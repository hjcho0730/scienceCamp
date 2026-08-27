<<<<<<< HEAD
<<<<<<< HEAD
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# main.py에 정의된 변수 및 함수들을 임포트 (model, device, class_names, transform 등 활용)
from main import *

# ==========================================
# 1. 검증 데이터셋 설정 (validData 폴더)
# ==========================================
valid_dir = "validData"

# validData 폴더 안의 하위 폴더들을 기반으로 데이터셋 생성
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

valid_classes = valid_dataset.classes
print(f"📁 validData에서 발견된 전체 클래스/폴더 목록: {valid_classes}\n")

# ==========================================
# 2. 정답 비교가 가능한 데이터 vs noLabel 분리 평가 함수
# ==========================================
def evaluate_and_predict():
    model.eval()
    
    total_images = 0
    correct_predictions = 0
    
    # 'noLabel' 폴더가 존재하는지 확인하고 인덱스 파악
    nolabel_idx = -1
    if "noLabel" in valid_classes:
        nolabel_idx = valid_classes.index("noLabel")
        print("🔍 'noLabel' 폴더가 감지되었습니다. 이 폴더의 이미지들은 정답 비교 없이 예측 결과만 출력합니다.\n")

    class_correct = {cls: 0 for cls in valid_classes if cls != "noLabel"}
    class_total = {cls: 0 for cls in valid_classes if cls != "noLabel"}

    # noLabel 이미지들을 따로 담아두고 출력을 예쁘게 하기 위한 리스트
    nolabel_results = []

    print("🚀 데이터셋 평가 및 예측 시작...")
    with torch.no_grad():
        # 이미지 파일 경로를 함께 추적하기 위해 dataset의 samples 활용
        # valid_dataset.samples는 (이미지 경로, 클래스 인덱스) 튜플의 리스트입니다.
        for idx, (inputs, labels) in enumerate(tqdm(valid_loader, desc="Processing")):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted_train_indices = torch.max(probabilities, 1)

            for i in range(len(labels)):
                batch_global_idx = idx * valid_loader.batch_size + i
                img_path, true_label_idx = valid_dataset.samples[batch_global_idx]
                
                # 1. 모델이 예측한 학습 기준 클래스 이름
                train_pred_idx = predicted_train_indices[i].item()
                pred_class_name = class_names[train_pred_idx]
                confidence = confidences[i].item() * 100

                true_class_name = valid_classes[true_label_idx]

                # 2. 만약 'noLabel' 폴더 안의 이미지라면 정답 비교 안 함
                if true_label_idx == nolabel_idx:
                    nolabel_results.append({
                        "path": img_path,
                        "prediction": pred_class_name,
                        "confidence": confidence
                    })
                    continue

                # 3. 일반 정답 폴더인 경우 정확도 계산
                class_total[true_class_name] += 1
                total_images += 1

                if pred_class_name == true_class_name:
                    correct_predictions += 1
                    class_correct[true_class_name] += 1

    # ==========================================
    # 3. 결과 리포트 출력
    # ==========================================
    if total_images > 0:
        overall_accuracy = (correct_predictions / total_images) * 100
        print("\n" + "="*40)
        print(f"📊 정답 비교 가능 데이터 평가 리포트")
        print("="*40)
        print(f"▪ 평가한 이미지 수: {total_images}장")
        print(f"▪ 맞춘 이미지 수: {correct_predictions}장")
        print(f"▪ **전체 정확도 (Overall Accuracy): {overall_accuracy:.2f}%**")
        print("="*40)

        print("\n📈 [클래스별 상세 정확도]")
        for cls in valid_classes:
            if cls != "noLabel" and class_total[cls] > 0:
                cls_acc = (class_correct[cls] / class_total[cls]) * 100
                print(f" - {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]}장)")

    # ==========================================
    # 4. noLabel 폴더 예측 결과 출력
    # ==========================================
    if nolabel_results:
        print("\n" + "="*40)
        print(f"🏷️ 'noLabel' 폴더 이미지 예측 결과")
        print("="*40)
        for item in nolabel_results:
            file_name = os.path.basename(item["path"])
            print(f"📂 파일명: {file_name}")
            print(f" 🌿 예측 결과: {item['prediction']}")
            print(f" 🔍 확신도: {item['confidence']:.2f}%")
            print("-" * 30)
    else:
        if nolabel_idx != -1:
            print("\nℹ️ 'noLabel' 폴더 내에 이미지가 존재하지 않습니다.")

if __name__ == "__main__":
=======
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# main.py에 정의된 변수 및 함수들을 임포트 (model, device, class_names, transform 등 활용)
from main import *

# ==========================================
# 1. 검증 데이터셋 설정 (validData 폴더)
# ==========================================
valid_dir = "validData"

# validData 폴더 안의 하위 폴더들을 기반으로 데이터셋 생성
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

valid_classes = valid_dataset.classes
print(f"📁 validData에서 발견된 전체 클래스/폴더 목록: {valid_classes}\n")

# ==========================================
# 2. 정답 비교가 가능한 데이터 vs noLabel 분리 평가 함수
# ==========================================
def evaluate_and_predict():
    model.eval()
    
    total_images = 0
    correct_predictions = 0
    
    # 'noLabel' 폴더가 존재하는지 확인하고 인덱스 파악
    nolabel_idx = -1
    if "noLabel" in valid_classes:
        nolabel_idx = valid_classes.index("noLabel")
        print("🔍 'noLabel' 폴더가 감지되었습니다. 이 폴더의 이미지들은 정답 비교 없이 예측 결과만 출력합니다.\n")

    class_correct = {cls: 0 for cls in valid_classes if cls != "noLabel"}
    class_total = {cls: 0 for cls in valid_classes if cls != "noLabel"}

    # noLabel 이미지들을 따로 담아두고 출력을 예쁘게 하기 위한 리스트
    nolabel_results = []

    print("🚀 데이터셋 평가 및 예측 시작...")
    with torch.no_grad():
        # 이미지 파일 경로를 함께 추적하기 위해 dataset의 samples 활용
        # valid_dataset.samples는 (이미지 경로, 클래스 인덱스) 튜플의 리스트입니다.
        for idx, (inputs, labels) in enumerate(tqdm(valid_loader, desc="Processing")):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted_train_indices = torch.max(probabilities, 1)

            for i in range(len(labels)):
                batch_global_idx = idx * valid_loader.batch_size + i
                img_path, true_label_idx = valid_dataset.samples[batch_global_idx]
                
                # 1. 모델이 예측한 학습 기준 클래스 이름
                train_pred_idx = predicted_train_indices[i].item()
                pred_class_name = class_names[train_pred_idx]
                confidence = confidences[i].item() * 100

                true_class_name = valid_classes[true_label_idx]

                # 2. 만약 'noLabel' 폴더 안의 이미지라면 정답 비교 안 함
                if true_label_idx == nolabel_idx:
                    nolabel_results.append({
                        "path": img_path,
                        "prediction": pred_class_name,
                        "confidence": confidence
                    })
                    continue

                # 3. 일반 정답 폴더인 경우 정확도 계산
                class_total[true_class_name] += 1
                total_images += 1

                if pred_class_name == true_class_name:
                    correct_predictions += 1
                    class_correct[true_class_name] += 1

    # ==========================================
    # 3. 결과 리포트 출력
    # ==========================================
    if total_images > 0:
        overall_accuracy = (correct_predictions / total_images) * 100
        print("\n" + "="*40)
        print(f"📊 정답 비교 가능 데이터 평가 리포트")
        print("="*40)
        print(f"▪ 평가한 이미지 수: {total_images}장")
        print(f"▪ 맞춘 이미지 수: {correct_predictions}장")
        print(f"▪ **전체 정확도 (Overall Accuracy): {overall_accuracy:.2f}%**")
        print("="*40)

        print("\n📈 [클래스별 상세 정확도]")
        for cls in valid_classes:
            if cls != "noLabel" and class_total[cls] > 0:
                cls_acc = (class_correct[cls] / class_total[cls]) * 100
                print(f" - {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]}장)")

    # ==========================================
    # 4. noLabel 폴더 예측 결과 출력
    # ==========================================
    if nolabel_results:
        print("\n" + "="*40)
        print(f"🏷️ 'noLabel' 폴더 이미지 예측 결과")
        print("="*40)
        for item in nolabel_results:
            file_name = os.path.basename(item["path"])
            print(f"📂 파일명: {file_name}")
            print(f" 🌿 예측 결과: {item['prediction']}")
            print(f" 🔍 확신도: {item['confidence']:.2f}%")
            print("-" * 30)
    else:
        if nolabel_idx != -1:
            print("\nℹ️ 'noLabel' 폴더 내에 이미지가 존재하지 않습니다.")

if __name__ == "__main__":
>>>>>>> ffb38c92000ce000a2ba35f5a488c6cb06052060
=======
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# main.py에 정의된 변수 및 함수들을 임포트 (model, device, class_names, transform 등 활용)
from main import *

# ==========================================
# 1. 검증 데이터셋 설정 (validData 폴더)
# ==========================================
valid_dir = "validData"

# validData 폴더 안의 하위 폴더들을 기반으로 데이터셋 생성
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)

valid_classes = valid_dataset.classes
print(f"📁 validData에서 발견된 전체 클래스/폴더 목록: {valid_classes}\n")

# ==========================================
# 2. 정답 비교가 가능한 데이터 vs noLabel 분리 평가 함수
# ==========================================
def evaluate_and_predict():
    model.eval()
    
    total_images = 0
    correct_predictions = 0
    
    # 'noLabel' 폴더가 존재하는지 확인하고 인덱스 파악
    nolabel_idx = -1
    if "noLabel" in valid_classes:
        nolabel_idx = valid_classes.index("noLabel")
        print("🔍 'noLabel' 폴더가 감지되었습니다. 이 폴더의 이미지들은 정답 비교 없이 예측 결과만 출력합니다.\n")

    class_correct = {cls: 0 for cls in valid_classes if cls != "noLabel"}
    class_total = {cls: 0 for cls in valid_classes if cls != "noLabel"}

    # noLabel 이미지들을 따로 담아두고 출력을 예쁘게 하기 위한 리스트
    nolabel_results = []

    print("🚀 데이터셋 평가 및 예측 시작...")
    with torch.no_grad():
        # 이미지 파일 경로를 함께 추적하기 위해 dataset의 samples 활용
        # valid_dataset.samples는 (이미지 경로, 클래스 인덱스) 튜플의 리스트입니다.
        for idx, (inputs, labels) in enumerate(tqdm(valid_loader, desc="Processing")):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted_train_indices = torch.max(probabilities, 1)

            for i in range(len(labels)):
                batch_global_idx = idx * valid_loader.batch_size + i
                img_path, true_label_idx = valid_dataset.samples[batch_global_idx]
                
                # 1. 모델이 예측한 학습 기준 클래스 이름
                train_pred_idx = predicted_train_indices[i].item()
                pred_class_name = class_names[train_pred_idx]
                confidence = confidences[i].item() * 100

                true_class_name = valid_classes[true_label_idx]

                # 2. 만약 'noLabel' 폴더 안의 이미지라면 정답 비교 안 함
                if true_label_idx == nolabel_idx:
                    nolabel_results.append({
                        "path": img_path,
                        "prediction": pred_class_name,
                        "confidence": confidence
                    })
                    continue

                # 3. 일반 정답 폴더인 경우 정확도 계산
                class_total[true_class_name] += 1
                total_images += 1

                if pred_class_name == true_class_name:
                    correct_predictions += 1
                    class_correct[true_class_name] += 1

    # ==========================================
    # 3. 결과 리포트 출력
    # ==========================================
    if total_images > 0:
        overall_accuracy = (correct_predictions / total_images) * 100
        print("\n" + "="*40)
        print(f"📊 정답 비교 가능 데이터 평가 리포트")
        print("="*40)
        print(f"▪ 평가한 이미지 수: {total_images}장")
        print(f"▪ 맞춘 이미지 수: {correct_predictions}장")
        print(f"▪ **전체 정확도 (Overall Accuracy): {overall_accuracy:.2f}%**")
        print("="*40)

        print("\n📈 [클래스별 상세 정확도]")
        for cls in valid_classes:
            if cls != "noLabel" and class_total[cls] > 0:
                cls_acc = (class_correct[cls] / class_total[cls]) * 100
                print(f" - {cls}: {cls_acc:.2f}% ({class_correct[cls]}/{class_total[cls]}장)")

    # ==========================================
    # 4. noLabel 폴더 예측 결과 출력
    # ==========================================
    if nolabel_results:
        print("\n" + "="*40)
        print(f"🏷️ 'noLabel' 폴더 이미지 예측 결과")
        print("="*40)
        for item in nolabel_results:
            file_name = os.path.basename(item["path"])
            print(f"📂 파일명: {file_name}")
            print(f" 🌿 예측 결과: {item['prediction']}")
            print(f" 🔍 확신도: {item['confidence']:.2f}%")
            print("-" * 30)
    else:
        if nolabel_idx != -1:
            print("\nℹ️ 'noLabel' 폴더 내에 이미지가 존재하지 않습니다.")

if __name__ == "__main__":
>>>>>>> ffb38c92000ce000a2ba35f5a488c6cb06052060
    evaluate_and_predict()