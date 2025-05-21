🚀 Space Station Object Detection using YOLOv8
Detect critical tools in a space station environment using a YOLOv8 object detection model, trained on synthetic data generated with Duality AI’s Falcon digital twin.

🛰️ Project Overview
This project aims to accurately detect essential space station tools using synthetic data:

Toolbox

Fire Extinguisher

Oxygen Tank

The YOLOv8 model was trained with synthetic data under various conditions:

Changing lighting

Occlusions (e.g., partially hidden tools)

Multiple angles and perspectives

🧠 Model Highlights
Architecture: YOLOv8 (from Ultralytics)

Pretrained Weights: yolov8s.pt

Configuration: yolo_params.yaml

Classes: Defined in classes.txt

📁 Repository Structure

predict.py: Script for running inference on images or videos.

train.py.py: Script to train the YOLOv8 model.

visualize.py: Visualizes predictions and training results.

yolo_params.yaml: Contains YOLOv8 training parameters.

classes.txt: Lists the object classes for detection.

yolov8s.pt: Pretrained YOLOv8 model weights.

runs/detect/: Directory containing detection outputs.

.gitignore: Specifies files and directories to ignore in Git.

🛠️ Setup & Installation
Clone the Repository:


git clone https://github.com/SardarAnmolsingh/AI-Thon1.0--Project-Team-Rocket-.git
cd AI-Thon1.0--Project-Team-Rocket-
Create a Virtual Environment:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:


pip install -r requirements.txt
Note: Ensure that requirements.txt contains all necessary packages.

🧪 Usage
Training the Model
To train the YOLOv8 model:


python train.py.py
Ensure that your dataset is properly formatted and located in the appropriate directory.

Running Inference
To run inference on images or videos:


python predict.py --source path_to_input --weights yolov8s.pt
Replace path_to_input with the path to your image or video file.

The results will be saved in the runs/detect/ directory.

Visualizing Results
To visualize predictions:


python visualize.py --source path_to_input
📊 Results
Detection Accuracy: Achieved high precision in detecting tools under varying conditions.

Robustness: Model performs well despite changes in lighting, occlusions, and perspectives.

Include sample images or metrics here to showcase model performance.

🤝 Contributors
Anmol Singh - GitHub
Rishikesh-Jadhav  
Maazsyedm    
rebekah-bogdanoff


📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or suggestions, please open an issue or contact Anmol Singh.
