<img width="1913" height="532" alt="Screenshot_2026-05-27_16-42-25" src="https://github.com/user-attachments/assets/496679d1-a2db-4cc6-8ffe-7040b54a6a28" />

## YOLO:FLO

YOLO:FLO is a data-efficient framework for flood-depth estimation from flood images. The proposed approach integrates computer-vision model, deep learning, machine learning, and modified few-shot learning. 

Find the detailed annotated data on the Roboflow website inside [FloodRepo](https://app.roboflow.com/arnab03-pbbfg/projects?group=fcY6ewqIM0TaE6rUzrNv).

We have used [DINO](https://docs.lightly.ai/train/stable/methods/distillation.html), [YOLO](https://docs.ultralytics.com/models/yolo11/) and [RT-DETR-v2-L](https://docs.ultralytics.com/models/rtdetr/) models in the pipeline. 

<img width="1171" height="820" alt="output_20260527_163849" src="https://github.com/user-attachments/assets/bc059925-c01d-4bfd-8f97-35546123819d" />


<img width="1918" height="721" alt="Screenshot_2026-05-27_16-48-28" src="https://github.com/user-attachments/assets/50bfa052-541e-4e32-8f52-0d9e68fc4584" />

This output shows multiple flood levels in the image. The final flood level of the image is determined by the YOLO:FLO in the interface.

The YOLO:FLO can annotate flood levels in [video](https://github.com/user-attachments/assets/0a884c9f-772f-4fb2-86ea-61b5f8565a91) and show the final level.

##CLI

First copy the python files in a folder name YOLOFLO. Then run this in the folder to open the YOLO:FLO app.

```bash
conda activate your_env_name
pip install streamlit
```
Then run the app.
```bash
streamlit run app.py
```
Update the file locations in the log files.


