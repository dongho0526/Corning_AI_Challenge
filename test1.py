import streamlit as st
import os
import zipfile
import base64
import io
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from ultralytics import YOLO
import shutil
import tempfile

# YOLOv8 모델 로드
def load_yolov8_model(weight_path=None):
    if weight_path:
        model = YOLO(weight_path)
    else:
        model = YOLO('models/yolov8/yolov8.pt')  # 기본 YOLOv8 가중치 경로
    return model

# EfficientNet 모델 로드
def load_efficientnet_model(weight_path=None):
    model = models.efficientnet_b0(pretrained=True)
    if weight_path:
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 객체 감지 및 분류 함수
def detect_and_classify(image, yolov8_model, efficientnet_model):
    results = yolov8_model.predict(np.array(image))

    # 바운딩 박스 좌표 추출
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    # 감지된 객체별로 클래스 분류 수행
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[:4]
        cropped_img = image.crop((xmin, ymin, xmax, ymax))

        # 전처리
        input_tensor = transform(cropped_img).unsqueeze(0)

        # EfficientNet 분류
        with torch.no_grad():
            outputs = efficientnet_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()

        # 바운딩 박스 그리기 및 클래스 라벨 표시
        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), str(class_id), fill="red")

    return image

def load_model(model_name, weight_file):
    if model_name == 'yolov5':
        return load_yolov5_model(weight_file)
    elif model_name == 'yolov8':
        return load_yolov8_model(weight_file)
    elif model_name == 'hybrid':
        yolov8_weight_path = weight_file
        efficientnet_weight_path = 'models/efficientnet/best_trained_efficientnet_weights_b7_2.pth'
        return (load_yolov8_model(yolov8_weight_path), load_efficientnet_model(efficientnet_weight_path))

def classify_images(model, model_name, uploaded_files):
    results = {class_name: [] for class_name in ['Defect1', 'Defect2', 'Defect3', 'Defect4', 'Defect5', 'Other']}
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            if model_name == 'hybrid':
                yolov8_model, efficientnet_model = model
                img = detect_and_classify(img, yolov8_model, efficientnet_model)
                img_base64 = pil_to_base64(img)
                results['Other'].append(img_base64)  # 모든 결과는 'Other' 카테고리에 추가
            else:
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
    return results

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
                results = st.session_state.results
                for class_name, images in results.items():
                    column = columns[class_names.index(class_name)]
                    for img_data in images:
                        img = Image.open(io.Bytes.IO(base64.b64decode(img_data)))
                        column.image(img, use_column_width=True)

    # Fine Tuning and Image Generation tabs go here...

def load_yolov5_model(weight_file):
    weight_dir = os.path.join(os.getcwd(), 'weights')
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = None
    if weight_file:
        weight_path = os.path.join(weight_dir, weight_file.name)
        with open(weight_path, "wb") as f:
            f.write(weight_file.getbuffer())
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom' if weight_path else 'yolov5s', path=weight_path, trust_repo=True)
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None
    return model

def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()

def draw_bounding_box(img, box, label):
    draw = ImageDraw.Draw(img)
    x_min, y_min, x_max, y_max = map(int, box)
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    draw.text((x_min, y_min), label, fill="red")

if __name__ == "__main__":
    if not os.getenv("STREAMLIT_RUNNING"):
        os.environ["STREAMLIT_RUNNING"] = "1"
        os.system("streamlit run " + __file__)
    else:
        main()
