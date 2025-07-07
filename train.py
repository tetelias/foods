from ultralytics import YOLO

def main() -> None:
    # Load a model
    model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="./dataset.yaml", 
                          epochs=50,
                          imgsz=640,
                          degrees=10.0,
                          hsv_h=0.03,
                          hsv_s=0.9,
                          hsv_v=0.6,
                          translate=0.3)


if __name__ == "__main__":
    main()
