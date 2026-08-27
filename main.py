<<<<<<< HEAD
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import vit_b_16
from PIL import Image

# ==========================================
# 0. 사용자 설정 (경로 및 모델 선택)
# ==========================================
# 테스트할 식물 질병 이미지 경로
image_path = "test_plant_leaf.jpg" 

# 저장해 둔 모델 가중치 파일 경로 (.pth)
model_path = "plant_disease_cnn_model.pth"  # 또는 "plant_disease_vit_model.pth"

# 사용할 모델 타입 선택 ('cnn' 또는 'vit')
model_type = "cnn"  # ViT 모델을 쓰셨다면 "vit"로 변경하세요.

# 학습 당시의 클래스 목록 (순서가 훈련 때와 정확히 일치해야 합니다!)
# 예시 클래스 목록입니다. 본인 데이터셋의 전체 클래스 이름으로 교체해 주세요.
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy',
]


num_classes = len(class_names)

# ==========================================
# 1. 디바이스 설정
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# ==========================================
# 2. 모델 구조 재생성 및 가중치 로드
# ==========================================
if model_type == "cnn":
    # EfficientNet-B0 구조 생성
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
elif model_type == "vit":
    # ViT-B/16 구조 생성
    model = vit_b_16(weights=None)
    in_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(in_features, num_classes)
else:
    raise ValueError("model_type은 'cnn' 또는 'vit'여야 합니다.")

# 저장된 가중치 불러오기
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval() # 평가 모드 전환
print("✅ 모델 로드 완료!")

# ==========================================
# 3. 이미지 전처리 정의 (학습할 때와 동일해야 함)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. 이미지 추론 (Prediction) 수행
# ==========================================
def predict_image_path(img_path):
    if not os.path.exists(img_path):
        print(f"❌ 오류: 이미지를 찾을 수 없습니다 -> {img_path}")
        return
    
    # 이미지 열기 및 전처리
    image = Image.open(img_path).convert('RGB')
    return predict_image(image)

def predict_image(img):

    input_tensor = transform(img).unsqueeze(0).to(device) # 배치 차원 추가 [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 소프트맥스를 거쳐 확률값으로 변환
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 가장 높은 확률을 가진 클래스 추출
        conf, predicted = torch.max(probabilities, 0)

    predicted_class = class_names[predicted.item()]
    confidence = conf.item() * 100

    print("\n--------------------")
    print(f"🌿 예측 결과: {predicted_class}")
    print(f"🔍 확신도(Confidence): {confidence:.2f}%")
    print("--------------------")

    return predicted_class, confidence

# 실행
if __name__ == "__main__":
=======
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import vit_b_16
from PIL import Image

# ==========================================
# 0. 사용자 설정 (경로 및 모델 선택)
# ==========================================
# 테스트할 식물 질병 이미지 경로
image_path = "test_plant_leaf.jpg" 

# 저장해 둔 모델 가중치 파일 경로 (.pth)
model_path = "plant_disease_cnn_model.pth"  # 또는 "plant_disease_vit_model.pth"

# 사용할 모델 타입 선택 ('cnn' 또는 'vit')
model_type = "cnn"  # ViT 모델을 쓰셨다면 "vit"로 변경하세요.

# 학습 당시의 클래스 목록 (순서가 훈련 때와 정확히 일치해야 합니다!)
# 예시 클래스 목록입니다. 본인 데이터셋의 전체 클래스 이름으로 교체해 주세요.
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy',
]


num_classes = len(class_names)

# ==========================================
# 1. 디바이스 설정
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# ==========================================
# 2. 모델 구조 재생성 및 가중치 로드
# ==========================================
if model_type == "cnn":
    # EfficientNet-B0 구조 생성
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
elif model_type == "vit":
    # ViT-B/16 구조 생성
    model = vit_b_16(weights=None)
    in_features = model.heads[0].in_features
    model.heads[0] = nn.Linear(in_features, num_classes)
else:
    raise ValueError("model_type은 'cnn' 또는 'vit'여야 합니다.")

# 저장된 가중치 불러오기
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval() # 평가 모드 전환
print("✅ 모델 로드 완료!")

# ==========================================
# 3. 이미지 전처리 정의 (학습할 때와 동일해야 함)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. 이미지 추론 (Prediction) 수행
# ==========================================
def predict_image_path(img_path):
    if not os.path.exists(img_path):
        print(f"❌ 오류: 이미지를 찾을 수 없습니다 -> {img_path}")
        return
    
    # 이미지 열기 및 전처리
    image = Image.open(img_path).convert('RGB')
    return predict_image(image)

def predict_image(img):

    input_tensor = transform(img).unsqueeze(0).to(device) # 배치 차원 추가 [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(input_tensor)
        
        # 소프트맥스를 거쳐 확률값으로 변환
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 가장 높은 확률을 가진 클래스 추출
        conf, predicted = torch.max(probabilities, 0)

    predicted_class = class_names[predicted.item()]
    confidence = conf.item() * 100

    print("\n--------------------")
    print(f"🌿 예측 결과: {predicted_class}")
    print(f"🔍 확신도(Confidence): {confidence:.2f}%")
    print("--------------------")

    return predicted_class, confidence

# 실행
if __name__ == "__main__":
>>>>>>> ffb38c92000ce000a2ba35f5a488c6cb06052060
    predict_image_path(image_path)