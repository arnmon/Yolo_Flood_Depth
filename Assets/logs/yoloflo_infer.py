import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from yoloflo_core import *

# Cell 10 — FINAL UPDATED INFERENCE CELL (ConvNeXt + NGBoost-compatible)
# Updated to use ConvNeXtBackbone (convnext_base) and NGBoost fusion checkpoints
from ultralytics import YOLO as UltralyticsYOLO
import joblib
from pathlib import Path
import numpy as np
import cv2
import torch
import pandas as pd
import os

# -----------------------
# LLM (load once globally)
# -----------------------
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("Loading Qwen3-VL model...")

LLM_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype="auto",
    device_map="auto"
)

LLM_PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

print("LLM loaded.")

# NGBoost (for unpickling models)
from ngboost import NGBClassifier

# -----------------------
# Safety / missing-global bootstrap
# -----------------------
# Ensure CLASS_HEIGHT_PRIORS_CM and DEFAULT_CLASS_HEIGHT are present (prevents NameError after kernel restart)
try:
    CLASS_HEIGHT_PRIORS_CM  # noqa: F821
except NameError:
    CLASS_HEIGHT_PRIORS_CM = {
        "person":170, "car":150, "bus":300, "truck":300,
        "bicycle":100, "motorcycle":110,
    }
try:
    DEFAULT_CLASS_HEIGHT  # noqa: F821
except NameError:
    DEFAULT_CLASS_HEIGHT = 150

# -----------------------
# Helper: load YOLO models
# -----------------------
def load_yolo_models():
    det = UltralyticsYOLO(YOLO_DET_PATH)
    pose = UltralyticsYOLO(YOLO_POSE_PATH) if os.path.exists(YOLO_POSE_PATH) else None
    wseg = UltralyticsYOLO(YOLO_WSEG_PATH) if os.path.exists(YOLO_WSEG_PATH) else None
    return det, pose, wseg

# ----------------------------------------
# Water segmentation (defensive)
# ----------------------------------------
def run_water_seg(wseg_model, image, conf=0.30):
    try:
        if wseg_model is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        res = wseg_model.predict(image, imgsz=1024, conf=conf, verbose=False)[0]
        if getattr(res, "masks", None) is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        mask = res.masks.data.cpu().numpy()
        if mask.ndim == 3:
            mask = (mask > 0.5).any(axis=0)
        else:
            mask = (mask > 0.5)
        return mask.astype(np.uint8)
    except Exception:
        return np.zeros(image.shape[:2], dtype=np.uint8)

# ----------------------------------------
# Pose extraction (defensive)
# ----------------------------------------
def extract_poses(pose_model, image, conf=0.25):
    if pose_model is None:
        return []
    try:
        res = pose_model.predict(image, imgsz=640, conf=conf, verbose=False)[0]
        if getattr(res, "keypoints", None) is None:
            return []
        arr = res.keypoints.data.cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
    except Exception:
        return []

# ----------------------------------------
# Base physics features (kept as before)
# ----------------------------------------
def compute_base_features(box, mask, img_w, img_h):
    x1,y1,x2,y2 = map(int,box)
    bw = max(1, x2-x1); bh = max(1, y2-y1)
    box_area = bw*bh

    submask = mask[y1:y2, x1:x2] if (y2>y1 and x2>x1) else np.zeros((0,0), dtype=np.uint8)
    water_pixels = int(submask.sum())
    water_area_frac = water_pixels / (box_area + 1e-9)

    ys = np.where(submask>0)[0]
    if len(ys)>0:
        top_local = ys.min() + y1
        water_height_frac = (y2 - top_local) / (bh + 1e-9)
    else:
        top_local = y2
        water_height_frac = 0.0

    if submask.size>0:
        sm = (submask*255).astype(np.uint8)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1,0, ksize=3)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0,1, ksize=3)
        grad = np.sqrt(gx*gx + gy*gy)
        sgf_mean = float(grad.mean())
        sgf_max = float(grad.max())
    else:
        sgf_mean, sgf_max = 0.0, 0.0

    water_top_y = int(top_local)

    return {
        "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
        "box_w": bw, "box_h": bh, "box_area": box_area,
        "water_pixels": water_pixels, "water_area_frac": water_area_frac,
        "water_top_y": water_top_y, "water_height_frac": water_height_frac,
        "sgf_mean": sgf_mean, "sgf_max": sgf_max, "WOMI": water_area_frac,
        "submergence_ratio": water_height_frac,
        "bbox_ratio": bh/(bw+1e-9),
        "water_frac_ratio": water_area_frac,
        "waterline_norm": water_top_y/(img_h+1e-9)
    }

# small safe nanmean used by pose helpers
def nanmean_safe(values, fallback):
    arr = []
    for v in values:
        if v is None:
            arr.append(np.nan)
        else:
            try:
                arr.append(float(v))
            except:
                arr.append(np.nan)
    out = np.nanmean(arr)
    if np.isnan(out):
        return fallback
    return out

# ----------------------------------------
# Person semantics (kept as before)
# ----------------------------------------
def person_semantics(pose, water_top_y, box):
    x1,y1,x2,y2 = map(int, box)
    bh = max(1, y2-y1)

    ankle_b = y1 + 0.95*bh
    knee_b  = y1 + 0.75*bh
    hip_b   = y1 + 0.55*bh
    chest_b = y1 + 0.35*bh
    head_b  = y1 + 0.15*bh

    mid_thigh_b    = y1 + 0.80*bh
    upper_thigh_b  = y1 + 0.70*bh
    lower_waist_b  = y1 + 0.60*bh

    ankle, knee, hip, chest, head = ankle_b, knee_b, hip_b, chest_b, head_b
    mid_thigh, upper_thigh, lower_waist = mid_thigh_b, upper_thigh_b, lower_waist_b

    if pose is not None:
        try:
            conf = pose[:,2]
        except Exception:
            conf = np.zeros(pose.shape[0])

        def gety(idx):
            try:
                if conf[idx] > 0.25:
                    return float(pose[idx,1])
            except Exception:
                return None

        ankle = nanmean_safe([gety(15), gety(16)], ankle_b)
        knee  = nanmean_safe([gety(13), gety(14)], knee_b)
        hip   = nanmean_safe([gety(11), gety(12)], hip_b)

        chest = nanmean_safe([gety(5), gety(6)], chest_b)
        head  = nanmean_safe([gety(0)], head_b)

    flags = {
        "person_sub_ankle": int(water_top_y > ankle),
        "person_sub_knee": int(water_top_y > knee),
        "person_sub_hip": int(water_top_y > hip),
        "person_sub_chest": int(water_top_y > chest),
        "person_sub_head": int(water_top_y > head),

        "person_sub_mid_thigh": int(water_top_y > mid_thigh),
        "person_sub_upper_thigh": int(water_top_y > upper_thigh),
        "person_sub_lower_waist": int(water_top_y > lower_waist),
    }

    denom = head - ankle
    if denom <= 0:
        depth = 0.0
        depth_fine = 0.0
    else:
        depth = (water_top_y - ankle) / denom
        depth_fine = (water_top_y - lower_waist) / denom
        depth = float(np.clip(depth, 0.0, 1.0))
        depth_fine = float(np.clip(depth_fine, 0.0, 1.0))

    flags["person_depth_norm"] = depth
    flags["person_depth_fine_ratio"] = depth_fine
    return flags

# ----------------------------------------
# Vehicle semantics (kept as before)
# ----------------------------------------
def vehicle_semantics(box, water_top_y, cname):
    cname = cname.lower()
    x1,y1,x2,y2 = map(int, box)
    bh = max(1, y2-y1)

    wheel = y1 + int(0.85*bh)
    mid   = y1 + int(0.50*bh)
    window= y1 + int(0.65*bh)
    roof  = y1

    flags = {}

    if "car" in cname:
        flags["car_sub_wheel"] = int(water_top_y > wheel)
        flags["car_sub_mid"] = int(water_top_y > mid)
        flags["car_sub_window"] = int(water_top_y > window)
        flags["car_sub_roof"] = int(water_top_y > roof)
        flags["car_sub_half_door"] = int(water_top_y > (y1 + int(0.40*bh)))
        flags["car_depth_norm"] = float(np.clip((water_top_y - wheel)/max(1, roof-wheel), 0, 1))

    if "truck" in cname or "bus" in cname:
        flags["truck_sub_wheel"] = int(water_top_y > wheel)
        flags["truck_sub_mid"] = int(water_top_y > mid)
        flags["truck_sub_window"] = int(water_top_y > window)
        flags["truck_sub_roof"] = int(water_top_y > roof)
        flags["truck_sub_hubcap"] = int(water_top_y > (y1 + int(0.92*bh)))
        flags["truck_depth_norm"] = float(np.clip((water_top_y - wheel)/max(1, roof-wheel), 0, 1))

    if "motorcycle" in cname or "motorbike" in cname:
        wheel2 = y1 + int(0.90*bh)
        seat2  = y1 + int(0.60*bh)
        handle = y1 + int(0.30*bh)
        flags["motorcycle_sub_wheel"] = int(water_top_y > wheel2)
        flags["motorcycle_sub_seat"] = int(water_top_y > seat2)
        flags["motorcycle_sub_handle"] = int(water_top_y > handle)
        flags["motorcycle_sub_engine_level"] = int(water_top_y > (y1 + int(0.75*bh)))
        flags["motorcycle_depth_norm"] = float(np.clip((water_top_y - wheel2)/max(1, handle-wheel2), 0, 1))

    if "bicycle" in cname or "cycle" in cname:
        wheel2 = y1 + int(0.90*bh)
        seat2  = y1 + int(0.60*bh)
        handle = y1 + int(0.30*bh)
        flags["bicycle_sub_wheel"] = int(water_top_y > wheel2)
        flags["bicycle_sub_seat"] = int(water_top_y > seat2)
        flags["bicycle_sub_handle"] = int(water_top_y > handle)
        flags["bicycle_depth_norm"] = float(np.clip((water_top_y - wheel2)/max(1, handle-wheel2), 0, 1))

    return flags

# ----------------------------------------
# Generic semantics
# ----------------------------------------
def generic_semantics(subratio):
    return {
        "obj_sub_20pct": int(subratio > 0.20),
        "obj_sub_50pct": int(subratio > 0.50),
        "obj_sub_80pct": int(subratio > 0.80)
    }

# safe estimate depth using priors (uses bootstrap globals)
def estimate_depth_cm(subratio, cname):
    try:
        sr = 0.0 if subratio is None else float(subratio)
        if np.isnan(sr): sr = 0.0
    except Exception:
        sr = 0.0
    cname_l = str(cname).strip().lower()
    ref = CLASS_HEIGHT_PRIORS_CM.get(cname_l, DEFAULT_CLASS_HEIGHT)
    return float(ref * max(0.0, min(1.0, sr))), float(ref)

# nearest pose (copied defensive)
def nearest_pose(box, poses):
    if len(poses)==0:
        return None
    x1,y1,x2,y2 = box
    cx=(x1+x2)/2; cy=(y1+y2)/2
    best=None; best_d=1e12
    for p in poses:
        try:
            xs=p[:,0]; ys=p[:,1]
            xs=xs[~np.isnan(xs)]; ys=ys[~np.isnan(ys)]
            if len(xs)==0:
                px,py=cx,cy
            else:
                px,py=xs.mean(), ys.mean()
        except Exception:
            px,py=cx,cy
        d=(px-cx)**2+(py-cy)**2
        if d<best_d:
            best_d=d; best=p
    return best

# =====================================================
# Build feature vector matching ALL_FEATURES
# =====================================================
def build_44_features(full_rgb, box, cname, wseg_model, pose_model):
    h,w = full_rgb.shape[:2]

    mask = run_water_seg(wseg_model, full_rgb)
    base = compute_base_features(box, mask, w, h)

    poses = extract_poses(pose_model, full_rgb)
    pose_for_obj = nearest_pose(box, poses) if len(poses)>0 else None

    sem = {}

    if any(t in cname for t in ["person","human","man","woman","child"]):
        sem.update(person_semantics(pose_for_obj, base["water_top_y"], box))

    if any(t in cname for t in ["car","bus","truck","motorbike","motorcycle","bicycle","cycle"]):
        sem.update(vehicle_semantics(box, base["water_top_y"], cname))

    sem.update(generic_semantics(base["submergence_ratio"]))

    est_cm, ref_cm = estimate_depth_cm(base["submergence_ratio"], cname)

    row = {}

    # base features
    for k,v in base.items(): row[k] = float(v)
    row["estimated_depth_cm"] = float(est_cm)
    row["ref_height_cm"] = float(ref_cm)
    row["physics_residual"] = 0.0

    # ensure all exist (ALL_FEATURES expected to be defined in earlier cells)
    for f in ALL_FEATURES:
        if f not in row:
            row[f] = 0.0

    # apply semantics
    for k,v in sem.items():
        row[k] = float(v)

    # build vector
    return np.array([row[k] for k in ALL_FEATURES], dtype=np.float32)

# =====================================================
# INFERENCE FUNCTION (ConvNeXt + NGBoost-aware)
# =====================================================
def infer_image(image_path, out_csv):
    meta_path = Path(OUT_DIR)/"folds_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing fold metadata at {meta_path} — run training first.")
    meta = joblib.load(meta_path)
    if len(meta)==0:
        raise RuntimeError("No fold metadata found. Train first.")

    # instantiate backbone as ConvNeXt (ensure ConvNeXtBackbone is defined in Cell 4)
    try:
        backbone = ConvNeXtBackbone(variant="convnext_base", pretrained=True, img_size=IMAGE_SIZE).to(DEVICE)
    except NameError:
        raise RuntimeError("ConvNeXtBackbone is not defined. Please ensure you replaced Cell 4 with the ConvNeXtBackbone implementation.")

    backbone.eval()

    det_model, pose_model, wseg_model = load_yolo_models()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = img_bgr[:,:,::-1]

    det_res = det_model.predict(img_rgb, imgsz=1024, conf=0.25)[0]
    names = det_model.model.names

    rows = []

    # Load fold models (defensive). We create per-fold fusion objects consistent with saved ckpt.
    fold_models=[]
    for fo in meta:
        # load state dict first to infer emb_dim expected by fusion ckpt
        state = torch.load(fo["fusion_ckpt"], map_location=DEVICE)
        # try to infer in_features of first linear in mlp (our FusionMLP uses mlp[0].weight)
        inferred_emb_dim = None
        try:
            w0 = None
            # common key names: 'mlp.0.weight' or similar - try heuristics
            if "mlp.0.weight" in state:
                w0 = state["mlp.0.weight"]
            else:
                for k in state.keys():
                    if k.endswith(".weight") and k.startswith("mlp."):
                        w0 = state[k]; break
            if w0 is not None:
                in_features = w0.shape[1]  # equals (hidden, emb_dim + NUM_CLASSES)
                inferred_emb_dim = int(in_features - NUM_CLASSES)
                if inferred_emb_dim <= 0:
                    inferred_emb_dim = None
        except Exception:
            inferred_emb_dim = None

        # if we couldn't infer, fallback to backbone.out_dim
        desired_emb_dim = inferred_emb_dim if inferred_emb_dim is not None else getattr(backbone, "out_dim", None)

        # create fusion with correct emb dim
        fusion = FusionMLP(emb_dim=desired_emb_dim, xgb_prob_dim=NUM_CLASSES).to(DEVICE)
        try:
            fusion.load_state_dict(state)
        except Exception as e:
            try:
                fusion.load_state_dict(state, strict=False)
            except Exception:
                raise RuntimeError(f"Failed to load fusion checkpoint {fo['fusion_ckpt']} into FusionMLP. Error: {e}")

        fusion.eval()

        # load NGBoost model and scaler
        ngb_model = joblib.load(fo["ngb_model"])
        scaler = joblib.load(fo["scaler_path"])

        # if backbone embedding dim differs from fusion expected emb dim, create a linear adapter
        adapter = None
        bb_out_dim = getattr(backbone, "out_dim", None)
        if bb_out_dim is not None and desired_emb_dim is not None and bb_out_dim != desired_emb_dim:
            adapter = torch.nn.Linear(bb_out_dim, desired_emb_dim).to(DEVICE)
            adapter.eval()
            print(f"[infer_image] created adapter: project {bb_out_dim} -> {desired_emb_dim} for fold {fo.get('fold', '??')}")

        fold_models.append({"fusion":fusion, "ngb":ngb_model, "scaler":scaler, "adapter":adapter})

    # Iterate boxes
    for i in range(len(det_res.boxes)):
        cls_id = int(det_res.boxes.cls[i].item())
        cname = names[cls_id].lower()

        if cname not in ["person","car","motorcycle","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = det_res.boxes.xyxy[i].cpu().numpy().astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = max(0, x2); y2 = max(0, y2)
        box=[x1,y1,x2,y2]

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size==0:
            img_t = val_transform(np.zeros((IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.uint8)).unsqueeze(0).to(DEVICE)
        else:
            img_t = val_transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = backbone(img_t)  # (1, bb_out_dim) or (1, D)
            if emb.dim() == 4:
                emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).reshape(emb.shape[0], -1)
            emb_np = emb.cpu().numpy()

        # Build full feature vector
        feat_vec = build_44_features(img_rgb, box, cname, wseg_model, pose_model)

        # ensemble prediction
        probs = np.zeros((1,NUM_CLASSES), dtype=np.float32)
        for fm in fold_models:
            Xs = fm["scaler"].transform(feat_vec.reshape(1,-1))
            xp = fm["ngb"].predict_proba(Xs)  # (1, NUM_CLASSES)

            emb_t = torch.from_numpy(emb_np).float().to(DEVICE)  # (1, bb_out_dim)
            if fm.get("adapter", None) is not None:
                with torch.no_grad():
                    emb_t = fm["adapter"](emb_t)

            xp_t = torch.from_numpy(xp).float().to(DEVICE)

            with torch.no_grad():
                logits = fm["fusion"](emb_t, xp_t)
                p = torch.softmax(logits, dim=1).cpu().numpy()

            probs += p

        probs /= max(1, len(fold_models))

        # --- apply meta-classifier filter for class-4 (if available) ---
        try:
            meta_clf_path = Path(OUT_DIR) / "meta_clf_class4.pkl"
            chosen_thr = 0.75  # keep in sync with your validation choice

            if meta_clf_path.exists():
                meta_clf = joblib.load(meta_clf_path)

                # fusion features
                fusion_prob4 = float(probs[0, 4])
                fusion_conf_val = float(probs.max())
                ngb_entropy_val = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))

                # proto_sim4: average similarity to prototypes (use first available prototypes file)
                proto_sim4_val = 0.0
                proto_files = []
                for fo in meta:
                    cand = Path(fo["fusion_ckpt"]).parent / f"prototypes_fold{fo['fold']}.npz"
                    if cand.exists():
                        proto_files.append(cand)
                if len(proto_files) > 0:
                    P = np.load(proto_files[0], allow_pickle=True)
                    prototypes = P["prototypes"].item()
                    if 4 in prototypes:
                        prot4 = np.array(prototypes[4], dtype=np.float32)
                        prot4 = prot4 / (np.linalg.norm(prot4) + 1e-12)
                        emb_vec = emb_np.reshape(-1)  # (D,)
                        emb_vec = emb_vec / (np.linalg.norm(emb_vec) + 1e-12)
                        proto_sim4_val = float(np.dot(emb_vec, prot4))

                # assemble meta feature vector (shape (1,4))
                X_meta_box = np.array([[fusion_prob4, fusion_conf_val, proto_sim4_val, ngb_entropy_val]], dtype=np.float32)
                meta_conf = meta_clf.predict_proba(X_meta_box)[:, 1][0]

                # if meta says this predicted-4 should be demoted, zero it out and renormalize
                if int(probs.argmax(axis=1)[0]) == 4 and meta_conf < chosen_thr:
                    probs[0, 4] = 0.0
                    s = probs.sum()
                    if s <= 0:
                        probs[0] = 1.0 / probs.shape[1]
                    else:
                        probs = probs / s
        except Exception:
            # defensive: if anything goes wrong, skip meta-filter and continue
            pass

        pred = int(probs.argmax(axis=1)[0]) if probs.size>0 else 0

        row = {
            "image_path": image_path,
            "class_name": cname,
            "box_x1": x1, "box_y1": y1,
            "box_x2": x2, "box_y2": y2,
            "predicted_level": pred
        }

        # Store full feature vector for debugging/analysis
        # Store feature vector
        for j, f in enumerate(ALL_FEATURES):
            row[f] = float(feat_vec[j])
        
        # Store probabilities ONCE
        for k in range(NUM_CLASSES):
            row[f"prob_L{k}"] = float(probs[0, k])
        
        row["pred_prob"] = float(probs[0, pred]) if 0 <= pred < NUM_CLASSES else 0.0
        
        # --- end snippet ---

        rows.append(row)

    df=pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

# Cell 9 — Visualize flood-level predictions with bounding boxes (clean version: no LLM, no final-line overlay)

import matplotlib.pyplot as plt

def visualize_inference(image_path, df_inf, save_path=None, show_confidence=False, max_display_size=1200):
    """
    Draw boxes from df_inf.
    df_inf must contain: box_x1, box_y1, box_x2, box_y2, class_name, predicted_level,
    and optionally pred_prob or prob_L* columns.
    """

    if df_inf is None or len(df_inf) == 0:
        raise ValueError("df_inf is empty or None — run infer_image() first.")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    h0, w0 = img.shape[:2]

    # simple color map for flood levels
    cmap = {}
    for lvl in range(0, 11):
        t = lvl / 10.0
        b = int((1.0 - t) * 0 + t * 0)
        g = int((1.0 - t) * 200 + t * 50)
        r = int((1.0 - t) * 0 + t * 200)
        cmap[lvl] = (b, g, r)

    img_out = img.copy()

    for idx, row in df_inf.iterrows():
        try:
            x1 = int(row["box_x1"])
            y1 = int(row["box_y1"])
            x2 = int(row["box_x2"])
            y2 = int(row["box_y2"])
            cls_name = str(row.get("class_name", "obj"))
            level = int(row.get("predicted_level", 0))
        except Exception:
            continue

        x1 = max(0, min(x1, w0-1))
        y1 = max(0, min(y1, h0-1))
        x2 = max(0, min(x2, w0-1))
        y2 = max(0, min(y2, h0-1))

        color = cmap.get(level, (0, 255, 0))
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        label = f"{cls_name} | L{level}"
        if show_confidence and "pred_prob" in df_inf.columns:
            try:
                prob = float(row.get("pred_prob", 0.0))
                label += f" {prob:.2f}"
            except Exception:
                pass

        text_y = max(12, y1 - 8)
        cv2.putText(
            img_out,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    # Resize for viewing
    H, W = img_out.shape[:2]
    scale = 1.0
    if max(H, W) > max_display_size:
        scale = max_display_size / max(H, W)
        img_disp = cv2.resize(img_out, (int(W*scale), int(H*scale)))
    else:
        img_disp = img_out

    # Convert BGR to RGB for display
    img_rgb = img_disp[:, :, ::-1]

    plt.figure(figsize=(12, 12 * (img_rgb.shape[0] / img_rgb.shape[1])))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    if save_path:
        cv2.imwrite(save_path, img_out)
        print("Saved annotated image:", save_path)

    return img_out

def run_llm_on_image(image_path):
    try:
        prompt = (
            "From this flood image with bounding boxes, determine the overall flood level by considering the most representative level by seeing the person, car, motorcycle, bicycle, truck, bus in the water. Return ONLY one value: L<number> Example: L0/L1/L2/L3/L4/L5/L6/L7/L8/L9/L10/L11 (Do not explain)."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = LLM_PROCESSOR.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(LLM_MODEL.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = LLM_MODEL.generate(**inputs, max_new_tokens=50)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = LLM_PROCESSOR.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        result = output_text[0].strip()

        print("LLM Output Raw:", result)  # DEBUG

        return result

    except Exception as e:
        print("LLM ERROR:", e)
        return "LLM_FAILED"


def run_inference(image_path):

    import time

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_path = f"{OUT_DIR}/inference_{timestamp}.csv"
    img_out  = f"{OUT_DIR}/output_{timestamp}.jpg"

    # ---- run inference ----
    df_inf = infer_image(
        image_path,
        out_csv=csv_path
    )

    # ---- visualize ----
    visualize_inference(
        image_path,
        df_inf,
        save_path=img_out
    )

    print("Running LLM...")

    # ---- LLM ----
    llm_output = run_llm_on_image(img_out)

    print("LLM finished")

    # ---- FINAL OUTPUT ----
    print("\n-----------------------------")
    print("Saved CSV:", csv_path)
    print("Saved Image:", img_out)
    print("Final Flood Level:", llm_output)
    print("-----------------------------")

    return llm_output

# ============================================================
# VIDEO SUPPORT (ADD BELOW YOUR EXISTING CODE — DO NOT MODIFY ABOVE)
# ============================================================

def is_video_file(path):
    return Path(path).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def extract_key_frames(video_path, out_dir="video_frames", num_frames=6):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, total_frames // num_frames)

    paths = []
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            path = os.path.join(out_dir, f"frame_{count}.jpg")
            cv2.imwrite(path, frame)
            paths.append(path)
            count += 1

            if count >= num_frames:
                break

        frame_id += 1

    cap.release()
    return paths


def infer_video_fast(video_path, output_video_path, output_csv_path, frame_skip=5):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    print("Loading models once...")

    det_model, pose_model, wseg_model = load_yolo_models()

    backbone = ConvNeXtBackbone(
        variant="convnext_base",
        pretrained=True,
        img_size=IMAGE_SIZE
    ).to(DEVICE)
    backbone.eval()

    meta = joblib.load(Path(OUT_DIR) / "folds_meta.pkl")

    fold_models = []
    for fo in meta:
        state = torch.load(fo["fusion_ckpt"], map_location=DEVICE)

        fusion = FusionMLP(
            emb_dim=getattr(backbone, "out_dim", 1024),
            xgb_prob_dim=NUM_CLASSES
        ).to(DEVICE)

        fusion.load_state_dict(state, strict=False)
        fusion.eval()

        fold_models.append({
            "fusion": fusion,
            "ngb": joblib.load(fo["ngb_model"]),
            "scaler": joblib.load(fo["scaler_path"])
        })

    frame_id = 0
    all_rows = []

    print("Running video inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        img_rgb = frame[:, :, ::-1]

        det_res = det_model.predict(img_rgb, imgsz=1024, conf=0.25)[0]
        names = det_model.model.names

        rows = []

        for i in range(len(det_res.boxes)):
            cls_id = int(det_res.boxes.cls[i].item())
            cname = names[cls_id].lower()

            if cname not in ["person","car","motorcycle","truck","bus","bicycle"]:
                continue

            x1,y1,x2,y2 = det_res.boxes.xyxy[i].cpu().numpy().astype(int)

            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            img_t = val_transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = backbone(img_t)
                if emb.dim() == 4:
                    emb = torch.nn.functional.adaptive_avg_pool2d(emb, 1).reshape(1, -1)
                emb_np = emb.cpu().numpy()

            feat_vec = build_44_features(img_rgb, [x1,y1,x2,y2], cname, wseg_model, pose_model)

            probs = np.zeros((1, NUM_CLASSES), dtype=np.float32)

            for fm in fold_models:
                Xs = fm["scaler"].transform(feat_vec.reshape(1,-1))
                xp = fm["ngb"].predict_proba(Xs)

                with torch.no_grad():
                    logits = fm["fusion"](
                        torch.from_numpy(emb_np).float().to(DEVICE),
                        torch.from_numpy(xp).float().to(DEVICE)
                    )
                    probs += torch.softmax(logits, dim=1).cpu().numpy()

            probs /= len(fold_models)
            pred = int(probs.argmax())

            rows.append({
                "frame_id": frame_id,
                "class_name": cname,
                "box_x1": x1, "box_y1": y1,
                "box_x2": x2, "box_y2": y2,
                "predicted_level": pred,
                "pred_prob": float(probs[0, pred])
            })

        df_frame = pd.DataFrame(rows)

        annotated = frame.copy()
        for _, r in df_frame.iterrows():
            x1,y1,x2,y2 = int(r.box_x1), int(r.box_y1), int(r.box_x2), int(r.box_y2)
            label = f"{r.class_name} L{r.predicted_level}"

            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(annotated)

        if len(df_frame) > 0:
            all_rows.append(df_frame)

        frame_id += 1

    cap.release()
    out.release()

    if all_rows:
        pd.concat(all_rows).to_csv(output_csv_path, index=False)

    return output_video_path


def run_video_inference(video_path):

    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    out_video = f"{OUT_DIR}/video_{timestamp}.mp4"
    out_csv   = f"{OUT_DIR}/video_{timestamp}.csv"

    infer_video_fast(video_path, out_video, out_csv)

    frames = extract_key_frames(out_video, f"{OUT_DIR}/frames_{timestamp}", 6)

    prompt = """From this video with bounding boxes, determine the overall flood level by considering the most representative level by seeing the person, car, motorcycle, bicycle, truck, bus in the water and each frame. Return ONLY one value: L<number> Example: L0/L1/L2/L3/L4/L5/L6/L7/L8/L9/L10/L11 (Do not explain)."""

    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt})

    inputs = LLM_PROCESSOR.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {k: v.to(LLM_MODEL.device) for k,v in inputs.items()}

    with torch.no_grad():
        out_ids = LLM_MODEL.generate(**inputs, max_new_tokens=50)

    out_ids = [o[len(i):] for i,o in zip(inputs["input_ids"], out_ids)]

    result = LLM_PROCESSOR.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    print("\n======================")
    print("Saved Video:", out_video)
    print("Final Flood Level:", result)
    print("======================")

    return result


# ============================================================
# OVERRIDE ORIGINAL FUNCTION (AUTO IMAGE / VIDEO)
# ============================================================

_run_inference_image = run_inference  # backup original


def run_inference(input_path):

    if is_video_file(input_path):
        print("VIDEO MODE")
        return run_video_inference(input_path)

    else:
        print("IMAGE MODE")
        return _run_inference_image(input_path)
