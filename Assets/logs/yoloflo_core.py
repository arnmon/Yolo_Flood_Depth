# Cell 1 - setup
import os, sys, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import cv2
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

# NGBoost (replaces XGB)
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from ngboost.scores import LogScore

# config
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (update for new 50+ features)
DATA_ROOT = "/home/arnab/Desktop/yolo/data/Flood_model/yolov11_refined_balanced/object_label_csv2ConvNeXT_NGBoost_FShL"
K_FOLD_DIR = f"{DATA_ROOT}/kfolds_train_only"
TEST_CSV = f"{DATA_ROOT}/test_balanced_20.csv"

# new output folder
OUT_DIR = f"{DATA_ROOT}/hybrid_resnet50_xgb_fusion_v1"
os.makedirs(OUT_DIR, exist_ok=True)

# YOLO weights for inference (feature computation)
YOLO_DET_PATH  = "yolo11m.pt"
YOLO_POSE_PATH = "yolo11m-pose.pt"
YOLO_WSEG_PATH = "/home/arnab/Desktop/yolo/data/Flood_model/UrbanFlood_WaterSeg/yolo11m-water-seg/weights/best.pt"

# model/hyperparams
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER = "cosine"   # "cosine","onecycle","step"
NUM_CLASSES = 11

ACT = nn.SiLU  # option A

# Cell 2 - feature list + utils

ALL_FEATURES = [
    # geometry
    "box_x1","box_y1","box_x2","box_y2","box_w","box_h","box_area",

    # water/base
    "water_pixels","water_area_frac","water_top_y","water_height_frac",
    "sgf_mean","sgf_max","WOMI","submergence_ratio","bbox_ratio",
    "water_frac_ratio","waterline_norm",

    # physics
    "estimated_depth_cm","ref_height_cm","physics_residual",

    # person semantics (renamed from human_*)
    "person_sub_ankle","person_sub_knee","person_sub_hip",
    "person_sub_chest","person_sub_head","person_depth_norm",

    #  NEW PERSON SEMANTICS
    "person_sub_mid_thigh",
    "person_sub_upper_thigh",
    "person_sub_lower_waist",
    "person_depth_fine_ratio",

    # generic submergence flags
    "obj_sub_20pct","obj_sub_50pct","obj_sub_80pct",

    # car semantics
    "car_sub_wheel","car_sub_mid","car_sub_window","car_sub_roof","car_depth_norm",

    #  NEW CAR SEMANTIC
    "car_sub_half_door",

    # bus semantics
    "bus_sub_wheel","bus_sub_mid","bus_sub_window","bus_sub_roof","bus_depth_norm",

    # truck semantics
    "truck_sub_wheel","truck_sub_mid","truck_sub_window","truck_sub_roof","truck_depth_norm",

    #  NEW TRUCK SEMANTIC
    "truck_sub_hubcap",

    # motorcycle semantics
    "motorcycle_sub_wheel","motorcycle_sub_seat","motorcycle_sub_handle","motorcycle_depth_norm",

    #  NEW MOTORCYCLE SEMANTIC
    "motorcycle_sub_engine_level",

    # bicycle semantics
    "bicycle_sub_wheel","bicycle_sub_seat","bicycle_sub_handle","bicycle_depth_norm"
]

def safe_load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        try:
            pil = Image.open(path).convert("RGB")
            arr = np.array(pil)[:, :, ::-1]
            return arr
        except:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return img[:, :, ::-1]

# image transforms
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2,0.02),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Cell 3 - dataset
class HybridCropDataset(Dataset):
    def __init__(self, csv_path, feature_cols, transform=None, scaler: StandardScaler=None):
        self.df = pd.read_csv(csv_path)

        # Ensure ALL_FEATURES exist (including new ones)
        for c in feature_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0

        self.feature_cols = feature_cols
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- load image --------
        img_path = row["image_path"]
        img = safe_load_image_rgb(img_path)
        h, w = img.shape[:2]

        # -------- crop by GT box --------
        x1 = int(row["box_x1"]); y1 = int(row["box_y1"])
        x2 = int(row["box_x2"]); y2 = int(row["box_y2"])

        # boundary clamp
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))

        if x2 <= x1 or y2 <= y1:
            crop = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            crop = cv2.resize(img[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE))

        # -------- transforms --------
        if self.transform is not None:
            img_t = self.transform(crop)
        else:
            img_t = val_transform(crop)

        # -------- load tabular features --------
        feats = np.array(
            [
                float(row[c]) if not pd.isna(row[c]) else 0.0
                for c in self.feature_cols
            ],
            dtype=np.float32
        )

        # scaling (safer reshape)
        if self.scaler is not None:
            feats = self.scaler.transform(feats.reshape(1, -1))[0]

        feats_t = torch.from_numpy(feats.astype(np.float32))

        # -------- label --------
        label = int(row["flood_level"])

        return img_t, feats_t, label

# Cell 4 - models (ConvNeXt backbone + FusionMLP)
import torch
import torch.nn as nn

# Try to import torchvision convnext variants and weights API; fallback if not present
try:
    from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
    try:
        # modern weights API names (if torchvision supports)
        from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
        _TV_HAS_WEIGHTS = True
    except Exception:
        _TV_HAS_WEIGHTS = False
except Exception:
    # torchvision may be older; we'll try to import via timm if available
    _TV_HAS_WEIGHTS = False
    convnext_tiny = convnext_small = convnext_base = convnext_large = None

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone wrapper.
    - variant: 'convnext_tiny','convnext_small','convnext_base','convnext_large'
    - removes classifier and returns pooled embedding vector per image
    - infers embedding dim automatically via a dummy forward
    """
    def __init__(self, variant="convnext_base", pretrained=True, img_size=IMAGE_SIZE, device="cpu"):
        super().__init__()
        self.img_size = img_size
        variant = variant.lower()
        supported = {
            "convnext_tiny": convnext_tiny,
            "convnext_small": convnext_small,
            "convnext_base": convnext_base,
            "convnext_large": convnext_large
        }

        model = None
        # 1) Try torchvision provided constructors first
        if variant in supported and supported[variant] is not None:
            ctor = supported[variant]
            if _TV_HAS_WEIGHTS and pretrained:
                # map to weights constant if available
                try:
                    weights_map = {
                        "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                        "convnext_small": ConvNeXt_Small_Weights.IMAGENET1K_V1,
                        "convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1,
                        "convnext_large": ConvNeXt_Large_Weights.IMAGENET1K_V1
                    }
                    w = weights_map.get(variant, None)
                    if w is not None:
                        model = ctor(weights=w)
                    else:
                        model = ctor(pretrained=pretrained)
                except Exception:
                    # fallback
                    model = ctor(pretrained=pretrained)
            else:
                try:
                    model = ctor(pretrained=pretrained)
                except Exception:
                    model = ctor()
        # 2) If torchvision ctor not present or failed, try timm
        if model is None:
            try:
                import timm
                # map to common timm names
                timm_names = {
                    "convnext_tiny": "convnext_tiny",
                    "convnext_small": "convnext_small",
                    "convnext_base": "convnext_base",
                    "convnext_large": "convnext_large"
                }
                name = timm_names.get(variant, "convnext_base")
                model = timm.create_model(name, pretrained=pretrained)
                print(f"[ConvNeXtBackbone] loaded timm model {name}")
            except Exception as e:
                raise RuntimeError(f"Failed to create ConvNeXt variant '{variant}' from torchvision or timm. Error: {e}")

        # remove classifier head robustly
        removed = False
        if hasattr(model, "classifier"):
            try:
                model.classifier = nn.Identity(); removed = True
            except Exception:
                pass
        if not removed:
            if hasattr(model, "head"):
                model.head = nn.Identity(); removed = True
            elif hasattr(model, "fc"):
                model.fc = nn.Identity(); removed = True
            elif hasattr(model, "reset_classifier"):
                try:
                    model.reset_classifier(0)
                    removed = True
                except Exception:
                    pass

        self.model = model
        self._infer_out_dim()

    def _infer_out_dim(self):
        self.model.eval()
        dummy = torch.randn(1, 3, self.img_size, self.img_size)
        try:
            with torch.no_grad():
                # prefer forward_features if available
                if hasattr(self.model, "forward_features"):
                    out = self.model.forward_features(dummy)
                else:
                    out = self.model(dummy)
        except Exception:
            # attempt using features + avgpool if available
            if hasattr(self.model, "features"):
                with torch.no_grad():
                    feats = self.model.features(dummy)
                    if hasattr(self.model, "avgpool"):
                        feats = self.model.avgpool(feats)
                    out = feats
            else:
                raise RuntimeError("Unable to run dummy forward on ConvNeXt model to infer embedding dim.")

        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)

        if not isinstance(out, torch.Tensor):
            raise RuntimeError("Unexpected output type from ConvNeXt forward: " + str(type(out)))

        self.out_dim = out.shape[1]
        print(f"[ConvNeXtBackbone] inferred out_dim = {self.out_dim}")

    def forward(self, x):
        out = None
        # prefer forward_features if present
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
        else:
            out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, 1).reshape(out.shape[0], -1)
        return out


# ---- FusionMLP (unchanged from your original) ----
class FusionMLP(nn.Module):
    """
    Hybrid head: concat(emb, XGB_probs) → MLP → logits
    """
    def __init__(self,
                 emb_dim=2048,
                 xgb_prob_dim=NUM_CLASSES,
                 hidden=512,
                 out_dim=NUM_CLASSES,
                 use_ordinal_branch=False):

        super().__init__()

        self.use_ordinal = use_ordinal_branch

        # small normalization helps fusion stability
        self.fuse_norm = nn.LayerNorm(emb_dim + xgb_prob_dim)

        # main fusion network
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + xgb_prob_dim, hidden),
            ACT(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.30),

            nn.Linear(hidden, hidden // 2),
            ACT(),
            nn.BatchNorm1d(hidden // 2),
            nn.Dropout(0.20),

            nn.Linear(hidden // 2, out_dim)
        )

        if self.use_ordinal:
            self.ordinal_head = nn.Sequential(
                nn.Linear(hidden // 2, (NUM_CLASSES - 1)),
            )
        else:
            self.ordinal_head = None

    def forward(self, emb, xgb_probs, return_features=False):
        x = torch.cat([emb, xgb_probs], dim=1)
        x = self.fuse_norm(x)
        out = self.mlp(x)

        if return_features:
            return x, out

        if self.use_ordinal:
            hidden_proj = self.mlp[:-1](x)
            ord_out = self.ordinal_head(hidden_proj)
            return out, ord_out

        return out

# cell5

def extract_embeddings(backbone, loader, device=DEVICE, return_paths=False):
    """
    Returns:
      embs: np.array (N, D)
      labels: np.array (N,)
      paths: list (N,) optional - image paths in same order as embeddings
    """
    backbone.eval()
    backbone.to(device)

    all_embs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            # loader yields (imgs, feats, labels) from HybridCropDataset
            imgs, feats, labs = batch
            imgs = imgs.to(device)
            emb = backbone(imgs)
            if emb.dim() == 4:
                emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).reshape(emb.shape[0], -1)

            all_embs.append(emb.cpu().numpy().astype(np.float32))
            all_labels.extend(labs.numpy().astype(np.int64).tolist())

            # try to collect paths if dataset exposes them via attribute `df` + index
            if hasattr(loader.dataset, "df"):
                # compute the slice of indices for this batch - best-effort: rely on DataLoader order
                # NOTE: this works if loader.shuffle=False; in train use shuffle=True so don't rely there
                pass

    embs = np.vstack(all_embs).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"Extracted embeddings: shape={embs.shape}, dtype={embs.dtype}")
    if return_paths:
        # best-effort: if dataset has df and no shuffling, return df.image_path in order
        try:
            paths = loader.dataset.df["image_path"].values.tolist()
        except Exception:
            paths = [None] * len(labels)
        return embs, labels, paths

    return embs, labels

# Cell 6 — UPDATED train_kfold_hybrid (Ordinal + Focal + Label Smooth) with NGBoost

# ===== Extra Losses =====

class FocalLoss(nn.Module):
    """ Standard Focal Loss for multi-class classification """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(logits, target)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class OrdinalLoss(nn.Module):
    """
    Penalizes large distance between prediction and true ordinal label.
    """
    def __init__(self, num_classes=11):
        super().__init__()
        self.num_classes = num_classes
        # fixed positions 0–10
        self.positions = torch.arange(num_classes).float().to(DEVICE)

    def forward(self, probs, target):
        # probs = softmax(logits)
        exp_val = (probs * self.positions).sum(dim=1)   # predicted numeric level
        return ((exp_val - target.float()) ** 2).mean()


def smooth_ce_loss(logits, target, smoothing=0.1):
    """
    Label smoothing CE loss.
    """
    num_classes = logits.size(1)
    true_dist = torch.zeros_like(logits)
    true_dist.fill_(smoothing / (num_classes - 1))
    true_dist.scatter_(1, target.unsqueeze(1), 1 - smoothing)
    return torch.mean(torch.sum(-true_dist * torch.log_softmax(logits, dim=1), dim=1))


# =====================================================
# ================ TRAIN K-FOLD ========================
# =====================================================

def train_kfold_hybrid():
    import joblib

    train_files = sorted(Path(K_FOLD_DIR).glob("kfold_train_*.csv"))
    val_files   = sorted(Path(K_FOLD_DIR).glob("kfold_val_*.csv"))
    assert len(train_files) == len(val_files) and len(train_files) > 0, "K-fold CSVs missing"

    fold_ckpts = []

    backbone = ConvNeXtBackbone(variant="convnext_base", pretrained=True, img_size=IMAGE_SIZE).to(DEVICE)
    backbone.eval()  # frozen backbone

    for fold, (tr_csv, val_csv) in enumerate(zip(train_files, val_files), start=1):
        print(f"\n=== FOLD {fold} ===")

        df_tr = pd.read_csv(tr_csv)
        df_val = pd.read_csv(val_csv)

        # ensure missing features exist
        for c in ALL_FEATURES:
            if c not in df_tr.columns: df_tr[c] = 0.0
            if c not in df_val.columns: df_val[c] = 0.0

        # fit scaler on train
        scaler = StandardScaler()
        X_tr = df_tr[ALL_FEATURES].fillna(0.0).values.astype(np.float32)
        scaler.fit(X_tr)
        # scaled train features (we will train NGBoost on these)
        X_tr_scaled = scaler.transform(X_tr)

        y_tr_np = df_tr["flood_level"].values.astype(np.int32)

        # datasets
        train_ds = HybridCropDataset(str(tr_csv), ALL_FEATURES, transform=train_transform, scaler=scaler)
        val_ds   = HybridCropDataset(str(val_csv), ALL_FEATURES, transform=val_transform, scaler=scaler)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # ---- NGBoost TRAIN (replaces XGB) -------------------
        # Explicit multi-class categorical Dist with NUM_CLASSES categories
        ngb_model = NGBClassifier(
            Dist=k_categorical(NUM_CLASSES),   # 11-class categorical
            Score=LogScore,
            n_estimators=400,
            learning_rate=0.05,
            natural_gradient=True,
            random_state=SEED,
            verbose=False
        )

        print("  [NGBoost] fitting on scaled tabular features...")
        ngb_model.fit(X_tr_scaled, y_tr_np)

        fold_dir = Path(OUT_DIR) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        ngb_path = fold_dir / "ngb_model.pkl"
        joblib.dump(ngb_model, ngb_path)

        # ---- EXTRACT EMBEDDINGS ----------
        emb_tr, y_tr = extract_embeddings(backbone, train_loader)
        emb_val, y_val = extract_embeddings(backbone, val_loader)
        # ---------- PROTOTYPE (few-shot) COMPUTE & SAVE ----------
        
        # Normalize embeddings per-sample
        emb_norm = emb_tr.copy()
        emb_norm /= (np.linalg.norm(emb_norm, axis=1, keepdims=True) + 1e-12)
        
        prototypes = {}
        intra_dists = {}
        
        for cls in range(NUM_CLASSES):
            idxs = np.where(y_tr == cls)[0]
            if len(idxs) == 0:
                prototypes[cls] = np.zeros((emb_norm.shape[1],), dtype=np.float32)
                intra_dists[cls] = np.array([0.0], dtype=np.float32)
                continue
            cls_embs = emb_norm[idxs]
            proto = cls_embs.mean(axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            prototypes[cls] = proto.astype(np.float32)
        
            # distances (cosine -> convert to 1 - cosine)
            sims = cls_embs.dot(proto)
            dists = 1.0 - sims
            intra_dists[cls] = dists.astype(np.float32)
        
        # per-class threshold = 95th percentile of intra-class distances
        d_thresh = {cls: float(np.percentile(intra_dists[cls], 95)) if len(intra_dists[cls])>0 else 1.0
                   for cls in range(NUM_CLASSES)}
        
        # NGBoost val uncertainty proxy: use entropy of val predict_proba
        X_val_scaled = scaler.transform(df_val[ALL_FEATURES].fillna(0.0).values.astype(np.float32))
        val_probs = ngb_model.predict_proba(X_val_scaled)
        # entropy per sample
        ent = -np.sum(np.clip(val_probs, 1e-12, 1.0) * np.log(np.clip(val_probs, 1e-12, 1.0)), axis=1)
        v_thresh = float(np.percentile(ent, 90))  # 90th percentile as threshold
        
        # save
        proto_path = fold_dir / f"prototypes_fold{fold}.npz"
        np.savez_compressed(
            proto_path,
            prototypes=prototypes,
            d_thresh=d_thresh,
            v_thresh=v_thresh
        )
        print(f"Saved prototypes & thresholds to {proto_path}")
        # ---------- end prototype save ----------


        # ---- NGBoost PROBS --------------------
        # train probs (for fusion train)
        prob_tr = ngb_model.predict_proba(X_tr_scaled)

        # val probs (for fusion val)
        X_val_raw = df_val[ALL_FEATURES].fillna(0.0).values.astype(np.float32)
        X_val_scaled = scaler.transform(X_val_raw)
        prob_val = ngb_model.predict_proba(X_val_scaled)

        # ---- Fusion dataset ---------------
        import torch.utils.data as tud
        train_f_ds = tud.TensorDataset(
            torch.from_numpy(emb_tr).float(),
            torch.from_numpy(prob_tr).float(),
            torch.from_numpy(y_tr).long()
        )
        val_f_ds = tud.TensorDataset(
            torch.from_numpy(emb_val).float(),
            torch.from_numpy(prob_val).float(),
            torch.from_numpy(y_val).long()
        )

        train_f_loader = DataLoader(train_f_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_f_loader   = DataLoader(val_f_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # ---- Fusion MLP --------------------
        fusion = FusionMLP(emb_dim=emb_tr.shape[1], xgb_prob_dim=NUM_CLASSES).to(DEVICE)

        opt = optim.RAdam(fusion.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        # Loss modules
        focal_loss_fn = FocalLoss(gamma=2.0)
        ord_loss_fn   = OrdinalLoss(num_classes=NUM_CLASSES)

        best_f1 = -1
        fusion_ckpt = fold_dir / "fusion_best.pth"
        scaler_path = fold_dir / "scaler.pkl"

        # save scaler with joblib (SAFE)
        joblib.dump(scaler, scaler_path)

        # ================= TRAINING LOOP =================
        for epoch in range(1, EPOCHS + 1):
            fusion.train()
            t_loss = 0
            t_preds, t_trues = [], []

            for eb, xb, lbl in train_f_loader:
                eb, xb, lbl = eb.to(DEVICE), xb.to(DEVICE), lbl.to(DEVICE)
                opt.zero_grad()

                logits = fusion(eb, xb)
                probs = torch.softmax(logits, dim=1)

                # ---- Combined loss ----
                ce = smooth_ce_loss(logits, lbl, smoothing=0.1)
                focal = focal_loss_fn(logits, lbl)
                ordl = ord_loss_fn(probs, lbl)

                loss = ce + 0.25 * focal + 0.5 * ordl

                loss.backward()
                opt.step()

                t_loss += loss.item() * eb.size(0)
                t_preds.extend(logits.argmax(1).cpu().numpy())
                t_trues.extend(lbl.cpu().numpy())
                scheduler.step()

            tr_acc = accuracy_score(t_trues, t_preds)

            # validation
            fusion.eval()
            v_preds, v_trues = [], []
            epoch_rmse = []
            
            with torch.no_grad():
                for eb, xb, lbl in val_f_loader:
                    eb, xb = eb.to(DEVICE), xb.to(DEVICE)
                    logits = fusion(eb, xb)
                    v_preds.extend(logits.argmax(1).cpu().numpy())
                    v_trues.extend(lbl.numpy())

                    probs = torch.softmax(logits, dim=1)
                    levels = torch.arange(NUM_CLASSES, device=probs.device).float()
                    exp_pred = (probs * levels).sum(dim=1)
            
                    rmse_batch = torch.sqrt(
                        torch.mean((exp_pred.cpu() - lbl.float()) ** 2)
                    ).item()
            
                    epoch_rmse.append(rmse_batch)
            
            # -------- AFTER validation loop --------
            mean_rmse = float(np.mean(epoch_rmse))   # <<< ADD THIS LINE
            
            # EXISTING LOGGER (DO NOT MOVE)
            log_epoch_metrics(epoch, fold, np.array(v_trues), np.array(v_preds), fold_dir)
            
            # -------- ADD RMSE LOGGING JUST AFTER --------
            EPOCH_METRICS[(fold, "rmse")].append({
                "epoch": epoch,
                "rmse": mean_rmse
            })
            joblib.dump(dict(EPOCH_METRICS), fold_dir / "epoch_metrics.pkl")
            
            # existing metrics
            val_acc = accuracy_score(v_trues, v_preds)
            val_f1 = f1_score(v_trues, v_preds, average="weighted")
            
            print(
                f"Fold {fold} Epoch {epoch} "
                f"tr_acc={tr_acc:.4f} "
                f"val_acc={val_acc:.4f} "
                f"val_f1={val_f1:.4f} "
                f"rmse={mean_rmse:.3f}"
            )
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(fusion.state_dict(), fusion_ckpt)
            

        # store metadata
        fold_ckpts.append({
            "fold": fold,
            "fusion_ckpt": str(fusion_ckpt),
            "ngb_model": str(ngb_path),
            "scaler_path": str(scaler_path)
        })

    # FINAL meta file (SAFE)
    meta_path = Path(OUT_DIR) / "folds_meta.pkl"
    joblib.dump(fold_ckpts, meta_path)
    print("Saved fold metadata:", meta_path)

    return fold_ckpts

# Cell 7 — UPDATED ensemble_eval_test (conservative mid-class adjustments + meta-filter)
def ensemble_eval_test():
    import joblib
    from math import ceil
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np

    meta_path = Path(OUT_DIR) / "folds_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}, run training first.")

    fold_infos = joblib.load(meta_path)
    print(f"Loaded {len(fold_infos)} fold metadata.")

    # ================================
    # Load test CSV
    # ================================
    df_test = pd.read_csv(TEST_CSV)
    for c in ALL_FEATURES:
        if c not in df_test.columns:
            df_test[c] = 0.0

    # ================================
    # Backbone for embeddings
    # ================================
    backbone = ConvNeXtBackbone(variant="convnext_base", pretrained=True, img_size=IMAGE_SIZE).to(DEVICE)
    backbone.eval()

    # dummy scaler for dataset loading (we use per-fold scalers later)
    scaler_dummy = StandardScaler()
    scaler_dummy.mean_  = np.zeros(len(ALL_FEATURES))
    scaler_dummy.scale_ = np.ones(len(ALL_FEATURES))

    test_ds = HybridCropDataset(TEST_CSV, ALL_FEATURES, 
                                transform=val_transform, 
                                scaler=scaler_dummy)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=4)

    # ================================
    # Extract embeddings once
    # ================================
    emb_list, y_list = [], []
    with torch.no_grad():
        for imgs, feats, labels in tqdm(test_loader, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            emb = backbone(imgs)
            if emb.dim() == 4:
                emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).reshape(emb.shape[0], -1)
            emb_list.append(emb.cpu().numpy())
            y_list.extend(labels.numpy())

    emb_test = np.vstack(emb_list)      # shape (N, D)
    y_test   = np.array(y_list)         # true labels
    X_test_raw = df_test[ALL_FEATURES].values.astype(np.float32)

    N = len(df_test)
    probs_accum = np.zeros((N, NUM_CLASSES), dtype=np.float32)

    # helper: proto scoring with per-class temps (ordinal smoothing)
    def proto_scores_for_batch_with_temps(emb_batch, prototypes, temp_per_class):
        emb_n = emb_batch / (np.linalg.norm(emb_batch, axis=1, keepdims=True) + 1e-12)  # (B,D)
        prot_mat = np.stack([prototypes[c] for c in range(NUM_CLASSES)], axis=0)       # (C,D)
        sims = emb_n.dot(prot_mat.T)  # (B, C) cosine sims

        # apply per-class temperature scaling (treated as multiplicative scale)
        temps = np.array([temp_per_class.get(c, 10.0) for c in range(NUM_CLASSES)], dtype=np.float32)  # (C,)
        sims_scaled = sims * temps[None, :]  # broadcast multiply each column by its corresponding temp

        sims_scaled = sims_scaled - sims_scaled.max(axis=1, keepdims=True)
        ex = np.exp(sims_scaled)
        proto_probs = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
        max_sim = sims.max(axis=1)
        argmax = sims.argmax(axis=1)
        return proto_probs, max_sim, argmax

    # ================================
    # Process each fold
    # ================================
    for idx, fo in enumerate(fold_infos, start=1):
        print(f"\nProcessing fold {idx}/{len(fold_infos)} ...")

        # ---- Load scaler ----
        scaler = joblib.load(fo["scaler_path"])

        # ---- NGBoost predictions ----
        ngb_model = joblib.load(fo["ngb_model"])
        X_scaled = scaler.transform(X_test_raw)
        ngb_probs = ngb_model.predict_proba(X_scaled)   # shape (N, NUM_CLASSES)

        # ---- Load Fusion MLP ----
        fusion = FusionMLP(
            emb_dim=emb_test.shape[1],
            xgb_prob_dim=NUM_CLASSES
        ).to(DEVICE)

        state = torch.load(fo["fusion_ckpt"], map_location=DEVICE)
        fusion.load_state_dict(state)
        fusion.eval()

        # ---- Forward pass for fusion MLP outputs ----
        fold_probs = []
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE):
                e = torch.from_numpy(emb_test[i:i+BATCH_SIZE]).float().to(DEVICE)
                p = torch.from_numpy(ngb_probs[i:i+BATCH_SIZE]).float().to(DEVICE)

                logits = fusion(e, p)
                soft = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(soft)
        fold_probs = np.vstack(fold_probs)

        # ---- Attempt to load prototypes for this fold ----
        fo_fold = int(fo.get("fold", idx))
        candidate = Path(fo["fusion_ckpt"]).parent / f"prototypes_fold{fo_fold}.npz"
        prototypes = None; d_thresh = None; v_thresh = None
        if candidate.exists():
            P = np.load(candidate, allow_pickle=True)
            prototypes = P["prototypes"].item()
            d_thresh = P["d_thresh"].item()
            v_thresh = float(P["v_thresh"])
            print(f"  Loaded prototypes from {candidate}")
        else:
            candidate2 = Path(fo["ngb_model"]).parent / f"prototypes_fold{fo_fold}.npz"
            if Path(candidate2).exists():
                P = np.load(candidate2, allow_pickle=True)
                prototypes = P["prototypes"].item()
                d_thresh = P["d_thresh"].item()
                v_thresh = float(P["v_thresh"])
                print(f"  Loaded prototypes from {candidate2}")

        # ---- If prototypes exist, compute proto_probs and gating; else fall back to fusion only ----
        if prototypes is not None:
            # --- adjust thresholds for mid classes (mild tightening) ---
            d_thresh_adj = {int(k): float(v) for k, v in d_thresh.items()}
            # conservative scaling: class4 -> 0.92, class3 & 5 -> 0.95
            for c, scale in [(4, 0.92), (3, 0.95), (5, 0.95)]:
                if c in d_thresh_adj:
                    d_thresh_adj[c] = d_thresh_adj[c] * scale

            # --- per-class temp for ordinal smoothing (mild) ---
            temp_per_class = {c: 10.0 for c in range(NUM_CLASSES)}
            temp_per_class[4] = 12.0
            temp_per_class[3] = 11.0
            temp_per_class[5] = 11.0

            # compute proto_probs, max_sim, argmax for all test samples in batches
            proto_probs_all = []
            max_sims_all = []
            argmax_all = []
            for i in range(0, N, BATCH_SIZE):
                e_batch = emb_test[i:i+BATCH_SIZE]
                proto_p, max_s, argm = proto_scores_for_batch_with_temps(e_batch, prototypes, temp_per_class)
                proto_probs_all.append(proto_p)
                max_sims_all.append(max_s)
                argmax_all.append(argm)
            proto_probs_all = np.vstack(proto_probs_all)
            max_sims_all = np.concatenate(max_sims_all)
            argmax_all = np.concatenate(argmax_all)

            # NGBoost entropy as uncertainty proxy
            ent_all = -np.sum(np.clip(ngb_probs,1e-12,1.0) * np.log(np.clip(ngb_probs,1e-12,1.0)), axis=1)

            # gating: keep if (1 - max_sim) <= d_thresh_adj[class] AND ent <= v_thresh
            keep_mask = np.zeros(N, dtype=bool)
            for i in range(N):
                cls = int(argmax_all[i])
                sim = float(max_sims_all[i])
                dist = 1.0 - sim
                cls_thresh = float(d_thresh_adj.get(cls, 1.0))
                if (dist <= cls_thresh) and (ent_all[i] <= v_thresh):
                    keep_mask[i] = True

            # ----- CLASS-CONDITIONAL FUSION WEIGHTS (CONSERVATIVE) -----
            # Only class 4 uses higher proto weight; classes 3 & 5 remain baseline
            proto_arg = proto_probs_all.argmax(axis=1)  # class index per sample from proto view
            fusion_conf = fold_probs.max(axis=1)  # confidence of fusion
            proto_sim = max_sims_all             # similarity to proto of chosen class
            
            w_proto_sample = np.full(N, 0.4, dtype=np.float32)
            
            # apply proto boost ONLY when fusion is uncertain AND proto is confident
            mask = (proto_arg == 4) & (fusion_conf < 0.55) & (proto_sim > 0.40)
            w_proto_sample[mask] = 0.55

            fold_combined = (w_proto_sample[:, None] * proto_probs_all) + ((1.0 - w_proto_sample)[:, None] * fold_probs)

            # for samples failing gating, reduce proto influence: prefer fusion but slightly smooth
            fold_combined[~keep_mask] = fold_probs[~keep_mask] * 0.7 + (1.0 / NUM_CLASSES) * 0.3

            fold_probs = fold_combined
        else:
            print("  Prototypes not found for this fold — using fusion-MLP probs only.")

        probs_accum += fold_probs

    # ================================
    # Average across folds
    # ================================
    probs_avg = probs_accum / len(fold_infos)

    # --- integrate meta-classifier filter for class 4 ---
    meta_clf_path = Path(OUT_DIR) / "meta_clf_class4.pkl"
    chosen_thr = 0.75  # threshold from validation sweep

    probs = probs_avg.copy()
    preds_base = probs.argmax(axis=1)
    fusion_prob4 = probs[:, 4]
    fusion_conf = probs.max(axis=1)
    ngb_entropy = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    # compute proto_sim4
    proto_sim4 = np.zeros(len(probs), dtype=np.float32)
    proto_files = []
    for fo in fold_infos:
        cand = Path(fo["fusion_ckpt"]).parent / f"prototypes_fold{fo['fold']}.npz"
        if cand.exists():
            proto_files.append(cand)
    if proto_files:
        P = np.load(proto_files[0], allow_pickle=True)
        prototypes = P["prototypes"].item()
        prot4 = np.array(prototypes[4], dtype=np.float32)
        prot4 = prot4 / (np.linalg.norm(prot4) + 1e-12)
        emb_n = emb_test / (np.linalg.norm(emb_test, axis=1, keepdims=True) + 1e-12)
        proto_sim4 = emb_n.dot(prot4)

    X_meta = np.vstack([fusion_prob4, fusion_conf, proto_sim4, ngb_entropy]).T

    # load or train meta classifier
    if meta_clf_path.exists():
        meta_clf = joblib.load(meta_clf_path)
    else:
        X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
            X_meta, (y_test==4).astype(int), np.arange(len(X_meta)),
            test_size=0.25, random_state=42, stratify=(y_test==4).astype(int)
        )
        meta_clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        meta_clf.fit(X_tr, y_tr)
        joblib.dump(meta_clf, meta_clf_path)
        print("Trained & saved meta classifier to:", meta_clf_path)

    meta_conf_all = meta_clf.predict_proba(X_meta)[:, 1]
    probs_proc = probs.copy()
    demote_mask_all = (preds_base == 4) & (meta_conf_all < chosen_thr)
    if demote_mask_all.any():
        probs_proc[demote_mask_all, 4] = 0.0
        row_sums = probs_proc.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
        probs_proc = probs_proc / row_sums

    preds = probs_proc.argmax(axis=1)

    acc  = accuracy_score(y_test, preds)
    f1   = f1_score(y_test, preds, average="weighted")
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_test, preds, average="weighted", zero_division=0)

    print("\n===== FINAL ENSEMBLE RESULTS =====")
    print("Accuracy :", acc)
    print("F1 Score :", f1)
    print("Precision:", prec)
    print("Recall   :", rec)

    return {
        "acc": acc, "f1": f1, "precision": prec, "recall": rec,
        "preds": preds, "probs": probs_proc, "trues": y_test
    }

# ================================
# Cell 8 — Epoch-wise metrics logger & plotter
# ================================

from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict

# Global container (persists during training)
EPOCH_METRICS = defaultdict(list)

def log_epoch_metrics(epoch, fold, y_true, y_pred, out_dir):
    """
    Logs per-class metrics for a given epoch and fold.
    """
    for cls in range(NUM_CLASSES):
        mask = (y_true == cls)

        acc = (y_pred[mask] == cls).mean() if mask.sum() > 0 else 0.0
        prec = precision_score(y_true, y_pred, labels=[cls],
                               average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, labels=[cls],
                            average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, labels=[cls],
                        average="macro", zero_division=0)

        EPOCH_METRICS[(fold, cls)].append({
            "epoch": epoch,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    # persist after every epoch (safe)
    save_path = Path(out_dir) / "epoch_metrics.pkl"
    joblib.dump(dict(EPOCH_METRICS), save_path)


# ================================
# Plotting function (run AFTER training)
# ================================
def plot_epoch_metrics(out_dir, fold=1):
    """
    Plots epoch-wise metrics for each class for a given fold.
    """
    path = Path(out_dir) / "epoch_metrics.pkl"
    assert path.exists(), "epoch_metrics.pkl not found"

    data = joblib.load(path)

    for cls in range(NUM_CLASSES):
        key = (fold, cls)
        if key not in data:
            continue

        df = pd.DataFrame(data[key])

        plt.figure()
        plt.plot(df["epoch"], df["f1"], label="F1")
        plt.plot(df["epoch"], df["precision"], label="Precision")
        plt.plot(df["epoch"], df["recall"], label="Recall")
        plt.plot(df["epoch"], df["accuracy"], label="Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"Fold {fold} — Flood Level {cls}")
        plt.legend()
        plt.grid(True)
        plt.show()

# ==========================================================
# FINAL NGBoost ↔ sklearn compatibility patch (FIT + PREDICT)
# ==========================================================
# Patches BOTH check_X_y and check_array inside NGBoost
# to support sklearn versions without `ensure_all_finite`.

import inspect
import sklearn.utils.validation as _sk_val
import ngboost.ngboost as _ngb_mod

# ---------- check_X_y patch ----------
if "ensure_all_finite" not in inspect.signature(_sk_val.check_X_y).parameters:

    _orig_check_X_y = _sk_val.check_X_y

    def _check_X_y_compat(
        X, y,
        accept_sparse=False,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_all_finite=None,
        ensure_2d=True,
        allow_nd=False,
        multi_output=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        y_numeric=False,
        estimator=None,
    ):
        if ensure_all_finite is not None:
            force_all_finite = ensure_all_finite

        return _orig_check_X_y(
            X, y,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            order=order,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            multi_output=multi_output,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            y_numeric=y_numeric,
            estimator=estimator,
        )

    _sk_val.check_X_y = _check_X_y_compat
    _ngb_mod.check_X_y = _check_X_y_compat


# ---------- check_array patch ----------
if "ensure_all_finite" not in inspect.signature(_sk_val.check_array).parameters:

    _orig_check_array = _sk_val.check_array

    def _check_array_compat(
        array,
        accept_sparse=False,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_all_finite=None,
        ensure_2d=True,
        allow_nd=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        estimator=None,
        input_name="",
    ):
        if ensure_all_finite is not None:
            force_all_finite = ensure_all_finite

        return _orig_check_array(
            array,
            accept_sparse=accept_sparse,
            accept_large_sparse=accept_large_sparse,
            dtype=dtype,
            order=order,
            copy=copy,
            force_all_finite=force_all_finite,
            ensure_2d=ensure_2d,
            allow_nd=allow_nd,
            ensure_min_samples=ensure_min_samples,
            ensure_min_features=ensure_min_features,
            estimator=estimator,
            input_name=input_name,
        )

    _sk_val.check_array = _check_array_compat
    _ngb_mod.check_array = _check_array_compat


print("[PATCH] NGBoost check_X_y + check_array compatibility applied")

def run_yoloflo():
    return ensemble_eval_test()
