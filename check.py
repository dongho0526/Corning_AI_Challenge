import torch
print(torch.cuda.is_available())  # True가 출력되면 GPU 사용 가능
print(torch.cuda.current_device())  # 현재 사용 중인 GPU ID 출력
print(torch.cuda.get_device_name(0))  # 첫 번째 GPU의 이름 출력

#python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt --device 0
