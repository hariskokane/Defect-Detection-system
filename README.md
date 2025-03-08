# Defect Detection System

## Project Overview
This project is a defect detection system using the YOLOv8 model to identify defects in objects such as bottles. The system is designed for real-time detection and analysis.

## Features
- Uses YOLOv8 for object detection
- Real-time defect identification
- Model training and evaluation capabilities
- Configurable dataset and model settings

## Project Structure
```
Defect_detection/
│── best.pt                 # Trained model weights
│── last.pt                 # Last training checkpoint
│── yolov8l.pt              # Pre-trained YOLOv8 model
│── detector.py             # Script for running defect detection
│── train.py                # Script for training the model
│── data.yaml               # Dataset configuration
│── requirements.txt        # Dependencies
│── Times New Roman.ttf     # Font file (possibly for GUI or annotation)
```

## Installation
### Prerequisites
Ensure you have Python installed. Recommended version: Python 3.8+

### Install Dependencies
Run the following command to install the required packages:
```sh
pip install -r requirements.txt
```

## Usage
### Running Detection
To run real-time defect detection, execute:
```sh
python detector.py
```

### Training the Model
To train the model with custom data, run:
```sh
python train.py
```
Ensure that `data.yaml` is correctly configured with dataset paths.

## Configuration
Modify `data.yaml` to define dataset paths, class labels, and other parameters for training.

## Model Weights
- `best.pt`: The best-performing trained model.
- `last.pt`: The latest checkpoint from training.
- `yolov8l.pt`: The pre-trained YOLOv8 model.

## Contributing
Feel free to contribute to this project by improving detection accuracy, adding new features, or optimizing performance.
