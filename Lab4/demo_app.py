from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="ANN Demo", page_icon="🔢", layout="wide")

APP_DIR = Path(__file__).resolve().parent

st.title("🔢 ANN Image Classification Demo")
st.caption("Tải ảnh lên để hệ thống tự dự đoán ngay trên giao diện.")

with st.sidebar:
    st.markdown("### Chọn model")
    model_type = st.selectbox(
        "Model type",
        ["MNIST (Digits)", "Cat vs Dog"],
    )
    st.markdown("### Hướng dẫn")
    st.write("1. Chọn model phù hợp với ảnh.")
    st.write("2. Tải ảnh JPG/PNG lên.")
    st.write("3. Kết quả sẽ tự hiện ra.")

# ============================================================
# TODO: Thay đổi đường dẫn model tại đây
# ============================================================
MODEL_PATHS = {
    "MNIST (Digits)": APP_DIR / "mnist_ann_best.pth",
    "Cat vs Dog": APP_DIR / "catdog_ann_best.pth",
}

CLASS_NAMES = {
    "MNIST (Digits)": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "Cat vs Dog": ["cat", "dog"],
}

IMG_SIZES = {
    "MNIST (Digits)": 28,
    "Cat vs Dog": 64,
}

MEAN_STD = {
    "MNIST (Digits)": ((0.1307,), (0.3081,)),
    "Cat vs Dog": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


class MNISTANN(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class CatDogANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


@st.cache_resource
def load_model(model_type):
    model_path = MODEL_PATHS[model_type]
    img_size = IMG_SIZES[model_type]
    
    if model_type == "MNIST (Digits)":
        model = MNISTANN(input_size=784, num_classes=10)
    else:
        input_size = img_size * img_size * 3
        model = CatDogANN(input_size=input_size)
    
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(image, model, model_type):
    img_size = IMG_SIZES[model_type]
    mean, std = MEAN_STD[model_type]
    
    if model_type == "Cat vs Dog":
        if image.mode != "RGB":
            image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        if image.mode != "L":
            image = image.convert("L")
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        
        if model_type == "Cat vs Dog":
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0
            result = CLASS_NAMES[model_type][pred]
            confidence = prob if pred == 1 else 1 - prob
        else:
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            result = CLASS_NAMES[model_type][pred]
            confidence = probs[0, pred].item()
    
    return result, confidence


st.markdown("---")
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    st.markdown("### Tải hình ảnh lên")
    uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
    else:
        image = None
        st.info("Hãy tải một hình ảnh để bắt đầu.")

with right_col:
    st.markdown("### Kết quả dự đoán")

    if image is None:
        st.write("Kết quả sẽ hiển thị ở đây ngay sau khi bạn tải ảnh lên.")
    else:
        with st.spinner("Đang dự đoán..."):
            try:
                model = load_model(model_type)
                result, confidence = predict(image, model, model_type)

                st.success(f"Kết quả: {result}")
                st.metric("Độ tin cậy", f"{confidence * 100:.2f}%")

            except FileNotFoundError:
                st.error(f"Không tìm thấy model. Kiểm tra lại đường dẫn: {MODEL_PATHS[model_type]}")
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

st.markdown("---")
st.caption("Demo ANN thuần (PyTorch) - Không sử dụng CNN hay pretrained models")