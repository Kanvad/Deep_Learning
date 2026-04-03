import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="ANN Demo", page_icon="🔢")

st.title("🔢 ANN Image Classification Demo")

st.markdown("### Chọn loại model:")
model_type = st.selectbox(
    "Model type",
    ["MNIST (Digits)", "Cat vs Dog"]
)

# ============================================================
# TODO: Thay đổi đường dẫn model tại đây
# ============================================================
MODEL_PATHS = {
    "MNIST (Digits)": "Lab4/mnist_ann_best.pth",
    "Cat vs Dog": "Lab4/catdog_ann_best.pth",
}

CLASS_NAMES = {
    "MNIST (Digits)": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "Cat vs Dog": ["cat", "dog"],
}

IMG_SIZES = {
    "MNIST (Digits)": 28,
    "Cat vs Dog": 64,
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


def load_model(model_type):
    model_path = MODEL_PATHS[model_type]
    img_size = IMG_SIZES[model_type]
    
    if model_type == "MNIST (Digits)":
        model = MNISTANN(input_size=784, num_classes=10)
    else:
        input_size = img_size * img_size * 3
        model = CatDogANN(input_size=input_size)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(image, model, model_type):
    img_size = IMG_SIZES[model_type]
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3) if model_type == "Cat vs Dog" else transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
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
st.markdown("### Tải hình ảnh lên:")

uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hình đã tải lên", use_container_width=True)
    
    if st.button("Dự đoán"):
        with st.spinner("Đang dự đoán..."):
            try:
                model = load_model(model_type)
                result, confidence = predict(image, model, model_type)
                
                st.success(f"🎯 Kết quả: **{result}**")
                st.info(f"📊 Độ chính xác: {confidence*100:.2f}%")
                
            except FileNotFoundError:
                st.error(f"❌ Không tìm thấy model! Vui lòng kiểm tra đường dẫn: {MODEL_PATHS[model_type]}")
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")

st.markdown("---")
st.caption("💡 Demo ANN thuần (PyTorch) - Không sử dụng CNN hay pretrained models")