from ultralytics import YOLO 
from pathlib import Path
import cv2 
import os
import yaml 


# Function to predict and save images and bounding boxes
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction
    results = model.predict(image_path, conf=0.5)

    result = results[0]

    # Draw boxes on the image
    img = result.plot()

    # Save the image with bounding boxes
    cv2.imwrite(str(output_path), img)

    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for i in range(len(result.boxes.cls)):
            cls_id = int(result.boxes.cls[i])
            x_center, y_center, width, height = result.boxes.xywh[i].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':

    # Set the current working directory
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    # Load YOLO parameters from YAML
    yaml_file = this_dir / 'yolo_params.yaml'
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test']) / 'images'
        else:
            print("No 'test' field found in yolo_params.yaml. Please add the test field with the path to test images.")
            exit()

    # Validate test image directory
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        exit()

    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        exit()

    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        exit()

    # Load trained YOLO model
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if (detect_path / f).is_dir() and f.startswith("train")]

    if len(train_folders) == 0:
        raise ValueError("No training folders found under runs/detect")

    # Let user select the training run if more than one is found
    idx = 0
    if len(train_folders) > 1:
        print("Select the training folder:")
        for i, folder in enumerate(train_folders):
            print(f"{i}: {folder}")
        try:
            choice = int(input("Enter the number: "))
            if choice < 0 or choice >= len(train_folders):
                raise ValueError
            idx = choice
        except ValueError:
            print("Invalid input. Exiting.")
            exit()

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        exit()

    model = YOLO(model_path)

    # Prepare output directories
    output_dir = this_dir / "predictions"
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Run predictions on each image
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    print(f"\n✅ Predicted images saved in: {images_output_dir}")
    print(f"✅ Bounding box labels saved in: {labels_output_dir}")
    print(f"✅ Model parameters loaded from: {yaml_file}")

    # Run validation (test evaluation) using the same data YAML
    metrics = model.val(data=str(yaml_file), split="test")
    print("\n📊 Validation Metrics:")
    print(metrics)
