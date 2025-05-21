# Space Station Object Detection using YOLOv8

This project focuses on detecting critical tools in a space station environment using YOLOv8, trained on high-quality synthetic data generated with Duality AI’s Falcon digital twin platform.

## 🚀 Overview

We developed a high-performance object detection model targeting essential space station tools:
- 🧰 Toolbox  
- 🧯 Fire Extinguisher  
- 🫙 Oxygen Tank  

Synthetic data was used to simulate various conditions:
- Lighting variations  
- Occlusions  
- Different camera angles and perspectives  

## 🧠 Model

We used [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection, fine-tuned on our synthetic dataset to achieve optimal performance.

### Key Metrics
- **Model**: YOLOv8
- **mAP@0.5**: High (detailed results in evaluation section)
- **Robustness**: Strong class-wise performance across conditions

## 📊 Evaluation

The project includes:
- mAP evaluation metrics  
- Class-wise performance reports  
- Confusion matrix  
- Failure case analysis  
- Inference visualizations  

## 📁 Project Structure

