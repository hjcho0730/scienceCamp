import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import kagglehub

# 1. 캐글 데이터셋 다운로드
path = kagglehub.dataset_download("emmarex/plantdisease")
print("다운로드된 기본 경로:", path)

# 2. 실제 질병 폴더들이 들어있는 내부 경로로 설정 ('PlantVillage' 또는 'plantvillage' 하위 폴더 지정)
# 대소문자에 유의하여 실제 폴더명과 일치시켜 줍니다.
data_dir = os.path.join(path, 'PlantVillage') 
if not os.path.exists(data_dir):
    data_dir = os.path.join(path, 'plantvillage')

print(f"최종 데이터 디렉토리: {data_dir}")

# 3. 데이터 전처리 및 증강 정의
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 4. 데이터셋 로드 (ImageFolder 이용)
full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"✅ 정상 인식된 클래스 개수: {num_classes}개")
print(f"✅ 클래스 목록 일부: {class_names[:5]} ...") # 너무 길어서 상위 5개만 출력

# Train / Validation 데이터셋 분할 (8:2 비율)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 5. 모델 설정 (EfficientNet-B0 전이학습)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 6. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * len(inputs)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

    return model

# 8. 학습 실행 (Epoch 수는 테스트 삼아 우선 3~5회 정도로 시작해 보세요)
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)

# 9. 모델 저장 코드
save_path = "plant_disease_cnn_model.pth"
torch.save(trained_model.state_dict(), save_path)
print(f"\n🎉 모델이 성공적으로 저장되었습니다: {save_path}")
