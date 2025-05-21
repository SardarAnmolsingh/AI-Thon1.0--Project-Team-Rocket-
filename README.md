# 🚀 Space Station Object Detection using YOLOv8

Detecting critical tools in a space station environment using a YOLOv8 object detection model, trained on synthetic data generated with **Duality AI’s Falcon digital twin**.

---

## 🛰️ Project Overview

This project aims to accurately detect essential space station tools using synthetic data:

- 🧰 **Toolbox**
- 🧯 **Fire Extinguisher**
- 🫙 **Oxygen Tank**

We trained a YOLOv8 model with synthetic data under various conditions:
- Changing **lighting**
- **Occlusions** (e.g., partially hidden tools)
- Multiple **angles and perspectives**

---

## 🧠 Model Highlights

- **Architecture:** YOLOv8 (from Ultralytics)
- **Data Source:** Duality AI’s Falcon (synthetic digital twin)
- **Training:** Fine-tuned on custom dataset with diverse conditions
- **Performance:**  
  ✅ High **mAP@0.5**  
  ✅ Strong **class-wise accuracy**  
  ✅ Robust across edge cases

---

## 📊 Evaluation Metrics

Included in the repo:
- 📈 mAP and precision/recall curves
- 📉 Confusion matrix
- 🔍 Failure case analysis
- 🖼️ Inference visualizations

---

## 📁 Folder Structure

📦 space-station-object-detection
├── data/ # Synthetic images and annotations
├── models/ # Trained YOLOv8 weights
├── notebooks/ # Training and evaluation notebooks
├── results/ # Visualizations and evaluation outputs
├── yolov8_config/ # Custom YOLOv8 training configs
├── requirements.txt
└── README.md
---

## 🛠️ Getting Started

### 1. Clone the Repository
git clone https://github.com/your-username/space-station-object-detection.git
cd space-station-object-detection

2. Install Dependencies
pip install -r requirements.txt

3. Run Inference
yolo task=detect mode=predict model=models/best.pt source=data/test_images/

🔧 Built With
🤖 YOLOv8 by Ultralytics

🛰️ Falcon Digital Twin by Duality AI

🐍 Python, OpenCV, NumPy, Matplotlib

📄 License
This project is licensed under the MIT License.

🙌 Contributions
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
