# 🛠️ Defect Detection System

## 🚀 Overview
This project is a cutting-edge **Defect Detection System** powered by **YOLOv8** for real-time object inspection. Designed to identify defects in bottles, it ensures quality control with **high accuracy and efficiency**.

---
## 🎯 Features
✅ **YOLOv8-powered detection** 🔍
✅ **Real-time defect identification** ⏳
✅ **Custom model training support** 📊
✅ **Configurable dataset & model settings** 🛠️

---
## 📂 Project Structure
```
Defect_detection/
├── 🏋 best.pt               # Best trained model weights
├── ⏳ last.pt               # Last training checkpoint
├── 📦 yolov8l.pt            # Pre-trained YOLOv8 model
├── 🚀 detector.py           # Runs real-time detection
├── 📜 train.py              # Training script
├── ⚙️ data.yaml             # Dataset configuration
├── 📌 requirements.txt      # Dependencies
├── 🔤 Times New Roman.ttf   # Font file (for GUI/annotations)
```

---
## 🛠️ Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Install Dependencies
```sh
pip install -r requirements.txt
```

---
## 🎬 Usage
### 🕵️ Run Real-Time Detection
```sh
python detector.py
```

### 🏋️ Train the Model
```sh
python train.py
```
Ensure `data.yaml` is properly configured with dataset paths.

---
## 🔧 Configuration
Modify `data.yaml` to define **dataset paths, class labels, and training parameters**.

---
## 📌 Model Weights
- 🎯 `best.pt`: Best-performing trained model.
- ⏳ `last.pt`: Latest checkpoint from training.
- 🏗️ `yolov8l.pt`: Pre-trained YOLOv8 model.

---
## 🤝 Contributing
Want to improve accuracy, add features, or optimize performance? **Pull requests are welcome!** 🚀


