import base64
import io
import streamlit as st
from PIL import Image, ImageDraw
import os
import zipfile
from ultralytics import YOLO
import numpy as np
import torch
import tempfile
import subprocess
import shutil
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tracemalloc
from efficientnet_pytorch import EfficientNet
import cv2

def main():
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        .container {
            border: 1px solid white;
            padding: 1px;
            height: 100%;
        }
        .blank {
            border: 1px solid transparent;
            padding: 13px;
            height: 100%;
        }
        .custom-button {
           background-color: #4CAF50;
           color: white;
           padding: 14px 20px;
           margin: 8px 0;
           border: none;
           cursor: pointer;
           width: 100%;
        }
        .stButton>button {
            width: 100%;
            padding: 25px 20px;
            font-size: 1px;
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    header = st.container()
    find_tab, fine_tuning_tab, image_generation_tab = st.tabs(["객체 찾기", "Fine Tuning", "Image Generation"])

    with find_tab:
        model_choose_infind, vae = st.columns([1, 1])
        upload_section, empty1, button_section = st.columns([5, 1, 4])
        left_column, right_column = st.columns([1, 6])

        with model_choose_infind:
            st.subheader("모델 선택")
            select_models = st.selectbox("사용할 모델", ['yolov5', 'yolov8', 'hybrid'], key='modelselect1')

        with upload_section:
            uploaded_files = st.file_uploader("파일을 업로드 하세요", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
            uploaded_pt = st.file_uploader("Weight를 업로드 하세요", type=["pt"])

        with button_section:
            st.markdown('<div class="blank">', unsafe_allow_html=True)
            classify_button = st.button('분류')
            st.markdown('<div class="blank">', unsafe_allow_html=True)
            download_button = st.button('Download')
            st.markdown('<div class="blank">', unsafe_allow_html=True)

            if classify_button and uploaded_files:
                try:
                    model = load_model(select_models, uploaded_pt)
                    results = classify_images(model, select_models, uploaded_files)
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"Error during classification: {e}")

            if download_button:
                if 'results' in st.session_state and st.session_state.results:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for class_name, images in st.session_state.results.items():
                            for idx, img_data in enumerate(images):
                                img = Image.open(io.BytesIO(base64.b64decode(img_data)))
                                img_byte_arr = io.BytesIO()
                                img.save(img_byte_arr, format='JPEG')
                                zip_file.writestr(f"{class_name}_{idx}.jpg", img_byte_arr.getvalue())
                    zip_buffer.seek(0)
                    st.download_button(label="Download ZIP", data=zip_buffer, file_name="classified_images.zip",
                                       mime="application/zip", key='download_zip')
                else:
                    st.warning("No classified images to download.")

        with left_column:
            st.markdown('<div class="container">', unsafe_allow_html=True)
            st.subheader("업로드 한 사진")
            if uploaded_files:
                for file in uploaded_files:
                    img = Image.open(file)
                    st.image(img, caption=file.name, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right_column:
            st.markdown('<div class="container">', unsafe_allow_html=True)
            defect1, defect2, defect3, defect4, defect5, rest = right_column.columns(6)
            defect1.subheader("Defect 1")
            defect2.subheader("Defect 2")
            defect3.subheader("Defect 3")
            defect4.subheader("Defect 4")
            defect5.subheader("Defect 5")
            rest.subheader("Other")

            class_names = ['Defect1', 'Defect2', 'Defect3', 'Defect4', 'Defect5', 'Other']
            columns = [defect1, defect2, defect3, defect4, defect5, rest]
            if 'results' in st.session_state:
                print("14")
                results = st.session_state.results
                print("15")
                for class_name, images in results.items():
                    print("16")
                    column = columns[class_names.index(class_name)]
                    print("17")
                    for img_data in images:
                        print("18")
                        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
                        print("19")
                        column.image(img, use_column_width=True)
                        print("20")

    with fine_tuning_tab:
        model_choose_infine = st.container()
        left_column, center_column, right_column = st.columns(3)

        with model_choose_infine:
            st.subheader("모델 선택")
            model_select = st.selectbox("사용할 모델", ['yolov5', 'yolov8'], key='modelselect2')

        with left_column:
            st.subheader("하이퍼파라미터 설정")
            epochs = st.number_input("에포크 수", min_value=1, max_value=100, value=100)
            batch_size = st.number_input("배치 크기", min_value=1, max_value=64, value=16)
            image_size = st.number_input("이미지 크기", min_value=1, max_value=1024, value=640)
            device = st.selectbox("사용할 장치", ["cpu", "cuda:0", "cuda:1"], index=1)
            uploaded_weights = st.file_uploader("가중치 파일을 업로드 하세요 (pt 파일)", type=["pt"])

        with center_column:
            st.subheader("데이터 증강 옵션")
            use_all_augmentations = st.checkbox("이미지 증강 모두 사용")

            augmentation_options = {
                "horizontal_flip": st.checkbox("수평 반전", value=use_all_augmentations),
                "vertical_flip": st.checkbox("수직 반전", value=use_all_augmentations),
                "rotation": st.checkbox("회전", value=use_all_augmentations),
                "color_jitter": st.checkbox("색상 변화", value=use_all_augmentations),
                "resized_crop": st.checkbox("크기 조정 및 크롭", value=use_all_augmentations),
                "gaussian_blur": st.checkbox("가우시안 블러", value=use_all_augmentations),
                "grayscale": st.checkbox("그레이스케일", value=use_all_augmentations),
                "invert": st.checkbox("색상 반전", value=use_all_augmentations),
                "posterize": st.checkbox("포스터화", value=use_all_augmentations),
                "affine": st.checkbox("어파인 변환", value=use_all_augmentations),
                "perspective": st.checkbox("원근 변환", value=use_all_augmentations)
            }

        with right_column:
            st.subheader("학습 데이터 업로드")
            train_data = st.file_uploader("학습 데이터를 업로드 하세요 (zip 파일)", type=["zip"])
            train_button = st.button("Training Start")
            download_button = st.button("Download Training Results")

            if train_button and model_select and uploaded_weights and train_data:
                try:
                    if model_select == 'yolov5':
                        data_dir = os.path.join(os.getcwd(), 'models', 'yolov5', 'dataset')
                        project_dir = os.path.join(os.getcwd(), 'models', 'yolov5', 'train')
                    elif model_select == 'yolov8':
                        data_dir = os.path.join(os.getcwd(), 'models', 'yolov8', 'dataset')
                        project_dir = os.path.join(os.getcwd(), 'models', 'yolov8', 'train')

                    # Create directories if they do not exist
                    os.makedirs(data_dir, exist_ok=True)
                    os.makedirs(project_dir, exist_ok=True)

                    # Clear the train directory
                    for root, dirs, files in os.walk(project_dir):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir in dirs:
                            shutil.rmtree(os.path.join(root, dir))

                    # Extract the uploaded zip file into the respective data directory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                        tmp.write(train_data.getvalue())
                        train_data_path = tmp.name

                    with zipfile.ZipFile(train_data_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)

                    # Find data.yaml path
                    data_yaml_path = None
                    for root, dirs, files in os.walk(data_dir):
                        if 'data.yaml' in files:
                            data_yaml_path = os.path.join(root, 'data.yaml')
                            break

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                        tmp.write(uploaded_weights.getvalue())
                        weights_path = tmp.name

                    if not data_yaml_path:
                        st.error("The uploaded zip file does not contain 'data.yaml'")
                    else:
                        # Apply augmentations if any
                        if any(augmentation_options.values()):
                            transform = get_transform(augmentation_options)
                            train_images_path = os.path.join(data_dir, "train", "images")
                            for img_name in os.listdir(train_images_path):
                                img_path = os.path.join(train_images_path, img_name)
                                img = Image.open(img_path).convert("RGB")
                                augmented_img = transform(img)
                                augmented_img_path = os.path.join(train_images_path, f"aug_{img_name}")
                                augmented_img = transforms.ToPILImage()(augmented_img)
                                augmented_img.save(augmented_img_path)

                        if model_select == 'yolov5':
                            with st.spinner('Training YOLOv5...'):
                                train_yolov5(weights_path, data_yaml_path, epochs, batch_size, image_size, device)
                        elif model_select == 'yolov8':
                            with st.spinner('Training YOLOv8...'):
                                train_yolov8(weights_path, data_yaml_path, epochs, batch_size, image_size, device)

                        st.session_state['project_dir'] = project_dir  # Save the project directory to session state

                except Exception as e:
                    st.error(f"Error during training: {e}")

            if download_button:
                if 'project_dir' in st.session_state:
                    project_dir = st.session_state['project_dir']
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for foldername, subfolders, filenames in os.walk(project_dir):
                            for filename in filenames:
                                file_path = os.path.join(foldername, filename)
                                arcname = os.path.relpath(file_path, project_dir)
                                zip_file.write(file_path, arcname)
                    zip_buffer.seek(0)
                    st.download_button(label="Download ZIP", data=zip_buffer, file_name="training_results.zip",
                                       mime="application/zip")
                else:
                    st.warning("No training results to download. Please train a model first.")

    with image_generation_tab:
        st.subheader("이미지 생성")

        upload_col, gen_col = st.columns([2, 1])

        with upload_col:
            gen_uploaded_zip = st.file_uploader("이미지와 라벨이 포함된 zip 파일을 업로드 하세요", type=["zip"], key="gen_uploaded_zip")

        with gen_col:
            st.markdown('<div class="blank">', unsafe_allow_html=True)
            gen_button = st.button("생성 시작", key="gen_button")

        if gen_button and gen_uploaded_zip:
            with tempfile.TemporaryDirectory() as tmpdirname:
                with tempfile.NamedTemporaryFile(delete=False, dir=tmpdirname, suffix=".zip") as tmp:
                    tmp.write(gen_uploaded_zip.getvalue())
                    gen_data_path = tmp.name

                with zipfile.ZipFile(gen_data_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdirname)

                image_dir = os.path.join(tmpdirname, 'images')  # 이미지를 추출할 폴더 경로
                if os.path.exists(image_dir):
                    received_files = os.listdir(image_dir)
                    print(f"받은 파일 목록: {received_files}")
                else:
                    st.error(f"디렉토리가 존재하지 않습니다: {image_dir}")

                dataset = ImageDataset(image_dir, transform=get_transform())
                dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = VAE().to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

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
                            print(
                                f'Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')

                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(dataloader.dataset):.4f}')

                output_dir = "vae_output"
                generate_images(model, device, dataloader, output_dir)
                st.success("이미지 생성이 완료되었습니다.")

        gen_download_button = st.button("Download", key="gen_download_button")

        if gen_download_button:
            with open("vae_output/vae_results.zip", "rb") as f:
                st.download_button(
                    label="Download ZIP",
                    data=f,
                    file_name="vae_results.zip",
                    mime="application/zip",
                    key="download_zip"
                )


def load_model(model_name, weight_file):
    weight_dir = os.path.join(os.getcwd(), 'weights')
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = None
    if weight_file:
        weight_path = os.path.join(weight_dir, weight_file.name)
        with open(weight_path, "wb") as f:
            f.write(weight_file.getbuffer())
    try:
        if model_name == 'yolov5':
            model = torch.hub.load('ultralytics/yolov5', 'custom' if weight_path else 'yolov5s', path=weight_path,
                                   trust_repo=True)
        elif model_name == 'yolov8':
            model = YOLO(weight_path if weight_path else 'yolov8s.pt')

        elif model_name == 'hybrid':
            model = YOLO(weight_path if weight_path else 'yolov8s.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model


def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def classify_images(model, model_name, uploaded_files):
    print(uploaded_files)
    results = {class_name: [] for class_name in ['Defect1', 'Defect2', 'Defect3', 'Defect4', 'Defect5', 'Other']}
    if model_name == 'yolov5' or model_name == 'yolov8':
        for uploaded_file in uploaded_files:
            try:
                img = Image.open(uploaded_file)
                img_data = np.array(img)
                if model_name == 'yolov5':
                    pred = model(img_data)
                    boxes = pred.xyxy[0].tolist()
                elif model_name == 'yolov8':
                    pred = model.predict(img_data)
                    boxes = pred[0].boxes

                if len(boxes) > 0:
                    for box in boxes:
                        if model_name == 'yolov5':
                            *xyxy, conf, cls = box
                        elif model_name == 'yolov8':
                            xyxy = box.xyxy[0]
                            conf = box.conf[0]
                            cls = box.cls[0]
                        class_name = model.names[int(cls)]
                        draw_bounding_box(img, xyxy, class_name)
                        img_base64 = pil_to_base64(img)
                        if class_name in results:
                            results[class_name].append(img_base64)
                        else:
                            results['Other'].append(img_base64)
                else:
                    img_base64 = pil_to_base64(img)
                    results['Other'].append(img_base64)
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")

    elif model_name == 'hybrid':
        print("1")
        effnet_model = EfficientNet.from_name('efficientnet-b7')
        print("2")
        effnet_weights_path = 'train_data/best_trained_efficientnet_weights_b7_2.pth'
        print("3")
        num_classes = 5  # 사용자 정의 클래스 수
        print("4")
        effnet_model._fc = torch.nn.Linear(effnet_model._fc.in_features, num_classes)
        print("5")
        effnet_model.load_state_dict(torch.load(effnet_weights_path))
        print("6")
        effnet_model.eval()
        print("7")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("8")
        for uploaded_file in uploaded_files:
            print("9")
            #crop_and_classify(uploaded_file, model, effnet_model, transform, results)
            CLASS_NAMES = {0: 'Defect1', 1: 'Defect2', 2: 'Defect3', 3: 'Defect4', 4: 'Defect5', 5: 'Other'}
            #image = cv2.imread(str(uploaded_file))
            #file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            #image = cv2.imdecode(file_bytes, 1)
            file_bytes = uploaded_file.read()
            print("a")
            image = Image.open(io.BytesIO(file_bytes))
            print("b")
            image_np = np.array(image)
            print("10")
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            print("11")
            pred = model(image_rgb)
            print("12")
            boxes = pred[0].boxes
            print("13")

            if len(boxes) > 0:  # 바운딩 박스가 존재하는 경우
                # 첫 번째 바운딩 박스 좌표 추출
                x1, y1, x2, y2 = map(int, boxes.xyxy[0])
                conf = boxes.conf[0].item()  # 신뢰도
                yolo_cls = int(boxes.cls[0].item())  # 클래스
                cropped_img = image_rgb[y1:y2, x1:x2]
                pil_img = Image.fromarray(cropped_img)
                input_tensor = transform(pil_img).unsqueeze(0)

                with torch.no_grad():
                    outputs = effnet_model(input_tensor)
                    _, predicted = outputs.max(1)

                class_id = predicted.item()
                class_name = CLASS_NAMES.get(class_id, 'Other')
                print(f"Object: Class ID {class_id} with confidence {conf:.2f}")
                print(f"yolo class id : {yolo_cls}")
                label = f'Class: {class_id+1}, Conf: {conf:.2f}'
                cv2.putText(cropped_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                labeled_pil_img = Image.fromarray(cropped_img)
                img_base64 = pil_to_base64(labeled_pil_img)
                results[class_name].append(img_base64)

    return results

# def crop_and_classify(uploaded_file, model, effnet_model, transform, results):
#     CLASS_NAMES = {0: 'Defect1', 1: 'Defect2', 2: 'Defect3', 3: 'Defect4', 4: 'Defect5', 5: 'Other'}
#     image = cv2.imread(str(uploaded_file))
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pred = model(image_rgb)
#     boxes = pred[0].boxes
#     print("9")
#
#     if len(boxes) > 0:  # 바운딩 박스가 존재하는 경우
#         # 첫 번째 바운딩 박스 좌표 추출
#         x1, y1, x2, y2 = map(int, boxes.xyxy[0])
#         conf = boxes.conf[0].item()  # 신뢰도
#         yolo_cls = int(boxes.cls[0].item())  # 클래스
#         cropped_img = image_rgb[y1:y2, x1:x2]
#         pil_img = Image.fromarray(cropped_img)
#         input_tensor = transform(pil_img).unsqueeze(0)
#
#         with torch.no_grad():
#             outputs = effnet_model(input_tensor)
#             _, predicted = outputs.max(1)
#
#         class_id = predicted.item()
#         class_name = CLASS_NAMES.get(class_id, 'Other')
#         print(f"Object: Class ID {class_id} with confidence {conf:.2f}")
#         print(f"yolo class id : {yolo_cls}")
#         label = f'Class: {class_id}, Conf: {conf:.2f}'
#         cv2.putText(cropped_img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#         labeled_pil_img = Image.fromarray(cropped_img)
#         results[class_name].append(labeled_pil_img)

def draw_bounding_box(img, box, label):
    draw = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = map(int, box)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    draw.text((x_min, y_min), label, fill="red")

def train_yolov5(weights_path, data_path, epochs, batch_size, img_size, device):
    project_dir = os.path.join(os.getcwd(), 'models', 'yolov5', 'train')
    command = [
        'python', os.path.join('models', 'yolov5', 'train.py'),
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', data_path,
        '--weights', weights_path,
        '--device', device,
        '--project', project_dir,
        '--name', 'exp'
    ]
    print(f"Running command: {' '.join(command)}")  # 명령어 출력
    subprocess.run(command)

def train_yolov8(weights_path, data_path, epochs, batch_size, img_size, device):
    project_dir = os.path.join(os.getcwd(), 'models', 'yolov8', 'train')
    command = [
        'yolo', 'task=detect', 'mode=train',
        f'model={weights_path}',
        f'data={data_path}',
        f'epochs={epochs}',
        f'imgsz={img_size}',
        f'batch={batch_size}',
        f'device={device}',
        f'project={project_dir}',
        f'name=exp'
    ]
    print(f"Running command: {' '.join(command)}")  # 명령어 출력
    subprocess.run(command)

def get_transform(augmentation_options):
    transform_list = []
    if augmentation_options["horizontal_flip"]:
        transform_list.append(transforms.RandomHorizontalFlip())
    if augmentation_options["vertical_flip"]:
        transform_list.append(transforms.RandomVerticalFlip())
    if augmentation_options["rotation"]:
        transform_list.append(transforms.RandomRotation(30))
    if augmentation_options["color_jitter"]:
        transform_list.append(transforms.ColorJitter())
    if augmentation_options["resized_crop"]:
        transform_list.append(transforms.RandomResizedCrop(224))
    if augmentation_options["gaussian_blur"]:
        transform_list.append(transforms.GaussianBlur(3))
    if augmentation_options["grayscale"]:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    if augmentation_options["invert"]:
        transform_list.append(transforms.RandomInvert())
    if augmentation_options["posterize"]:
        transform_list.append(transforms.RandomPosterize(bits=4))
    if augmentation_options["affine"]:
        transform_list.append(transforms.RandomAffine(degrees=30))
    if augmentation_options["perspective"]:
        transform_list.append(transforms.RandomPerspective())

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


# tracemalloc 활성화
tracemalloc.start()

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 이미지 생성 함수
def generate_images(model, device, data_loader, output_dir):
    model.eval()
    original_dir = os.path.join(output_dir, "original")
    reconstructed_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)

    with torch.no_grad():
        for data, filenames in data_loader:
            data = data.to(device)
            data = data * 0.5 + 0.5
            recon_batch, mu, logvar = model(data)
            data = data.cpu().permute(0, 2, 3, 1).numpy()
            recon_batch = recon_batch.cpu().permute(0, 2, 3, 1).numpy()

            for i in range(data.shape[0]):
                original_img = Image.fromarray((data[i] * 255).astype(np.uint8))
                recon_img = Image.fromarray((recon_batch[i] * 255).astype(np.uint8))
                original_img.save(os.path.join(original_dir, filenames[i]))
                recon_img.save(os.path.join(reconstructed_dir, filenames[i].replace('.jpg', '_reconstructed.jpg').replace('.jpeg', '_reconstructed.jpeg').replace('.png', '_reconstructed.png')))

    with zipfile.ZipFile(os.path.join(output_dir, "vae_results.zip"), 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    zipf.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), output_dir))

if __name__ == "__main__":
    if not os.getenv("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        os.system("streamlit run " + __file__)
    else:
        main()
