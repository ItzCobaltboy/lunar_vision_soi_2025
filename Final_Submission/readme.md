# 🌑 Lunar Vision  
### by **Box Box Engineers** 🏎️ (yes, it’s an F1 reference)

---

## 🚀 What’s This?

Lunar Vision is our project for detecting craters on Moon surface images.

We built two models:

- ✅ A YOLOv8-based detector — trained, tested, and zipped for your convenience
- 🔬 A custom mini YOLO-style model — designed from scratch for single-class detection

The goal? Find **craters**. Only craters. Moon's pretty chill like that.

---

## 🗂️ Folder Vibes

```
.
├── yolo_model/               ← YOLOv8 model setup
│   ├── yolo_weights.pt       ← Trained YOLOv8 weights
│   ├── Simple_YOLO.py        ← Inference script to generate YOLO-style .txt labels
│   ├── dataset_evaluator.py  ← Training script (if you feel like retraining)
│   └── labels.zip            ← Our YOLO model's output on test set
├── custom_model/             ← Our handcrafted YOLO-style model
│   └── custom_model.py       ← Model code + custom loss function
├── requirements.txt          ← Python dependencies
├── report.pdf                ← Our write-up (yeah, we went full LaTeX mode)
└── README.md                 ← This file
```

---

## 📦 YOLO Model (`yolo_model/`)

This one’s good to go:

- `dataset_evaluator.py`: run inference on test images, outputs YOLO-format `.txt` labels
- `Simple_YOLO.py`: for retraining the YOLO model (optional)
- `labels.zip`: zipped up inference output for your evaluation
- `yolo_weights.pt`: trained weights (YOLOv8N)

To run inference:

```bash
cd yolo_model
python dataset_evaluator.py
```

Make sure you’ve got your test images ready — `.txt` labels will be saved in `labels/`.

---

## 🧪 Custom Model (`custom_model/`)

This is our custom baby:

- Designed from scratch to be **small and fast**
- Only supports one class (craters)
- Includes a **custom loss** with:
  - Spatial penalization for bad box locations
  - CLou (Center Localization Uncertainty) to punish bad box sizes

BUT — we didn’t train it all the way. It needs **800–1000 epochs** to converge.

> 🛑 We ran out of compute. If you’ve got an HPU or magic cloud credits — please train it. We believe in it. 🙏

---

## 📄 Report?

Yep. There’s a `report.pdf` in the root if you want the formal-ish side of this thing.  
(It’s in LaTeX, don’t ask. We got carried away.)

---

## ⚙️ Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## 🙌 Team

We’re **Box Box Engineers**

- Did we overbuild? Probably.
- Did we have fun? Also yes.

Let us know if our custom model accidently works 🤧🤧🚀

