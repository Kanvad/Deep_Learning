import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="CNN Demo", page_icon="🌿")

st.title("🌿 CNN Image Classification Demo")

st.markdown("### Chọn loại model:")
model_type = st.selectbox(
    "Model type",
    ["Cat & Dog", "CIFAR-10", "PlantVillage"]
)

# ============================================================
# TODO: Thay đổi đường dẫn model tại đây
# ============================================================
MODEL_PATHS = {
    "Cat & Dog": "Lab6\catdog_pure_cnn_best.pth",
    "CIFAR-10": "Lab6\cifar10_pure_cnn_best.pth",
    "PlantVillage": "Lab6\plantvillage_pure_cnn_best.pth",
}

CLASS_NAMES = {
    "Cat & Dog": ["cat", "dog"],
    "CIFAR-10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "PlantVillage": [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", "Corn___Cercospora_leaf_spot",
        "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_(Two-spotted_spider_mite)", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ],
}

IMG_SIZES = {
    "Cat & Dog": 224,
    "CIFAR-10": 32,
    "PlantVillage": 128,
}

# Transform values
MEAN_STD = {
    "Cat & Dog": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "CIFAR-10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    "PlantVillage": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PlantVillageCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_type):
    model_path = MODEL_PATHS[model_type]
    
    if model_type == "Cat & Dog":
        model = CatDogCNN()
    elif model_type == "CIFAR-10":
        model = CIFAR10CNN(num_classes=10)
    else:
        model = PlantVillageCNN(num_classes=38)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(image, model, model_type):
    img_size = IMG_SIZES[model_type]
    mean, std = MEAN_STD[model_type]
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        
        if model_type == "Cat & Dog":
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
st.caption("💡 Demo CNN thuần (PyTorch) - Không sử dụng pretrained models")