from ultralytics import YOLO

def main():
    model = YOLO('yolov8l.pt')  
    try:
        model.train(
            data='C:/Users/haris/OneDrive/Desktop/122/data.yaml',  
            epochs=100,
            imgsz=640,
            batch=8,  
            device='0'
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()
