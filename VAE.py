import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import zipfile
import numpy as np
import tracemalloc

# tracemalloc 활성화
tracemalloc.start()

# 실행 시작 알림
print("VAE 코드 실행 시작")

# Custom Dataset 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 주요 실행 코드
if __name__ == "__main__":
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 받은 파일 목록 출력
    image_dir = 'vae_input/images'  # 이미지를 추출할 폴더 경로
    if os.path.exists(image_dir):
        received_files = os.listdir(image_dir)
        print(f"받은 파일 목록: {received_files}")
    else:
        print(f"디렉토리가 존재하지 않습니다: {image_dir}")

    # 데이터셋 및 데이터로더 생성
    dataset = ImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # VAE 모델 정의
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            # Encoder
            self.enc1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.fc1 = nn.Linear(64*32*32, 256)
            self.fc_mu = nn.Linear(256, 128)
            self.fc_logvar = nn.Linear(256, 128)
            # Decoder
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 64*32*32)
            self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.dec3 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        def encode(self, x):
            x = torch.relu(self.enc1(x))
            x = torch.relu(self.enc2(x))
            x = torch.relu(self.enc3(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc_mu(x), self.fc_logvar(x)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            x = torch.relu(self.fc2(z))
            x = torch.relu(self.fc3(x))
            x = x.view(x.size(0), 64, 32, 32)
            x = torch.relu(self.dec1(x))
            x = torch.relu(self.dec2(x))
            return torch.sigmoid(self.dec3(x))

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    # 손실 함수 정의
    def loss_function(recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # 모델 초기화
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)  # 데이터 GPU로 전송
            data = data * 0.5 + 0.5  # Normalize 된 데이터를 [0, 1] 범위로 복원
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}')

    # 테스트 이미지 생성
    model.eval()
    output_dir = "vae_output"
    original_dir = os.path.join(output_dir, "original")
    reconstructed_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    with torch.no_grad():
        for data, filenames in dataloader:
            data = data.to(device)  # 데이터 GPU로 전송
            data = data * 0.5 + 0.5  # Normalize 된 데이터를 [0, 1] 범위로 복원
            recon_batch, mu, logvar = model(data)
            data = data.cpu().permute(0, 2, 3, 1).numpy()
            recon_batch = recon_batch.cpu().permute(0, 2, 3, 1).numpy()

            for i in range(data.shape[0]):
                original_img = Image.fromarray((data[i] * 255).astype(np.uint8))
                recon_img = Image.fromarray((recon_batch[i] * 255).astype(np.uint8))
                original_img.save(os.path.join(original_dir, filenames[i]))
                recon_img.save(os.path.join(reconstructed_dir, filenames[i].replace('.jpg', '_reconstructed.jpg').replace('.jpeg', '_reconstructed.jpeg').replace('.png', '_reconstructed.png')))

    # 결과 파일을 zip으로 압축
    with zipfile.ZipFile(os.path.join(output_dir, "vae_results.zip"), 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    zipf.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), output_dir))

    # 실행 완료 알림
    print("VAE 코드 실행 완료")
    print(f"생성된 파일 목록: {os.listdir(output_dir)}")
