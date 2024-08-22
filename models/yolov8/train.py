from ultralytics import YOLO

def train():
    # YOLOv8 모델 불러오기
    model = YOLO('yolov8s.pt')

    # 학습 시작 전에 데이터 경로 출력
    data_path = 'data.yaml'
    print(f"Using dataset from: {data_path}")

    # 학습 시작
    model.train(data=data_path, epochs=50, imgsz=640, batch=16, device=0)

if __name__ == '__main__':
    train()
