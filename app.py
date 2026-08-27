import os
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import vit_b_16

# ==========================================
# 0. 페이지 설정 및 디자인
# ==========================================
st.set_page_config(
    page_title="식물 질병 AI 자동 라벨링 시스템", page_icon="🌿", layout="wide"
)

st.title("🌿 식물 질병 AI 분류 및 상세 검증 시스템")
st.write(
    "CNN 또는 ViT 모델을 활용해 식물 질병을 진단하고, 각 이미지별 판별 결과를"
    " 상세하게 열람할 수 있습니다."
)

# ==========================================
# 1. 설정 및 모델 로드 (사이드바)
# ==========================================
st.sidebar.header("⚙️ 모델 및 설정")

# model_path = st.sidebar.text_input(
#     "모델 가중치 경로 (.pth)", value="plant_disease_cnn_model.pth"
# )
model_type = st.sidebar.selectbox(
    "모델 타입 선택", options=["cnn", "vit"], index=0
)

# 학습 당시의 클래스 목록
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
num_classes = len(class_names)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"사용 중인 디바이스: **{device}**")


# 모델 로드 함수 (캐싱 적용)
@st.cache_resource
def load_model(m_type, m_path, n_classes):
  if not os.path.exists(m_path):
    return None, f"오류: 모델 파일('{m_path}')을 찾을 수 없습니다."

  try:
    if m_type == "cnn":
      model = models.efficientnet_b0(weights=None)
      num_ftrs = model.classifier[1].in_features
      model.classifier[1] = nn.Linear(num_ftrs, n_classes)
    elif m_type == "vit":
      model = vit_b_16(weights=None)
      in_features = model.heads[0].in_features
      model.heads[0] = nn.Linear(in_features, n_classes)
    else:
      raise ValueError("model_type은 'cnn' 또는 'vit'여야 합니다.")

    model.load_state_dict(torch.load(m_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, None
  except Exception as e:
    return None, f"모델 로드 중 오류 발생: {str(e)}"


# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 모델 로드 실행
with st.spinner("AI 모델을 불러오는 중입니다..."):
  model, err_msg = load_model(model_type, f"plant_disease_{model_type}_model.pth", num_classes)

if err_msg:
  st.sidebar.error(err_msg)
  st.stop()
else:
  st.sidebar.success("✅ 모델 로드 완료!")

# ==========================================
# 2. 탭 구성
# ==========================================
tab1, tab2 = st.tabs([
    "🔍 단일 이미지 라벨링 (웹 업로드)",
    "📂 validData 폴더 일괄 평가 및 결과 열람",
])

# ------------------------------------------
# [탭 1] 단일 이미지 업로드 및 추론
# ------------------------------------------
with tab1:
  st.subheader("단일 식물 잎 이미지 AI 라벨링")
  st.write(
      "테스트할 이미지 파일(JPG, PNG 등)을 업로드하면 모델이 즉시 질병을"
      " 진단합니다."
  )

  uploaded_file = st.file_uploader(
      "식물 잎 이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
      st.image(image, caption="업로드된 이미지", use_column_width=True)

    with col2:
      if st.button("🚀 AI 라벨링 실행", type="primary"):
        with st.spinner("분석 중입니다..."):
          input_tensor = transform(image).unsqueeze(0).to(device)
          with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            conf, predicted = torch.max(probabilities, 0)

          predicted_class = class_names[predicted.item()]
          confidence = conf.item() * 100

          st.markdown("### 📊 분석 결과")
          st.success(f"**예측된 질병(라벨):** `{predicted_class}`")
          st.metric(label="확신도 (Confidence)", value=f"{confidence:.2f}%")

          result_str = (
              f"파일명: {uploaded_file.name}\n예측 결과: {predicted_class}\n확신도:"
              f" {confidence:.2f}%\n"
          )
          st.download_button(
              label="📥 결과 텍스트 다운로드",
              data=result_str,
              file_name=f"result_{uploaded_file.name}.txt",
              mime="text/plain",
          )

# ------------------------------------------
# [탭 2] validData 폴더 일괄 평가 및 상세 열람
# ------------------------------------------
with tab2:
  st.subheader("validData 폴더 일괄 평가 및 이미지별 판별 내역 열람")
  st.write(
      "서버의 `validData` 내 이미지들을 읽어와 정답과 예측을 비교하고, 각"
      " 이미지가 어떻게 판별되었는지 상세 카드로 보여줍니다."
  )

  valid_dir = "validData"

  if not os.path.exists(valid_dir):
    st.warning(
        f"⚠️ 현재 경로에 `{valid_dir}` 폴더가 존재하지 않습니다. 폴더를"
        " 생성하고 하위에 테스트 이미지를 넣어주세요."
    )
  else:
    if st.button("📊 일괄 평가 및 상세 결과 보기", type="primary"):
      with st.spinner(
          "데이터셋을 불러오고 평가를 진행 중입니다. 잠시만 기다려주세요..."
      ):
        try:
          valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
          valid_loader = DataLoader(
              valid_dataset, batch_size=32, shuffle=False, num_workers=0
          )
          valid_classes = valid_dataset.classes

          total_images = 0
          correct_predictions = 0
          nolabel_idx = (
              valid_classes.index("noLabel")
              if "noLabel" in valid_classes
              else -1
          )

          class_correct = {cls: 0 for cls in valid_classes if cls != "noLabel"}
          class_total = {cls: 0 for cls in valid_classes if cls != "noLabel"}

          # 상세 판별 결과를 저장할 리스트
          detailed_results = []
          nolabel_results = []

          model.eval()
          with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
              inputs = inputs.to(device)
              labels = labels.to(device)

              outputs = model(inputs)
              probabilities = torch.nn.functional.softmax(outputs, dim=1)
              confidences, predicted_train_indices = torch.max(
                  probabilities, 1
              )

              for i in range(len(labels)):
                batch_global_idx = idx * valid_loader.batch_size + i
                img_path, true_label_idx = valid_dataset.samples[
                    batch_global_idx
                ]

                train_pred_idx = predicted_train_indices[i].item()
                pred_class_name = class_names[train_pred_idx]
                confidence = confidences[i].item() * 100
                true_class_name = valid_classes[true_label_idx]

                # noLabel인 경우 별도 분류
                if true_label_idx == nolabel_idx:
                  nolabel_results.append({
                      "path": img_path,
                      "prediction": pred_class_name,
                      "confidence": confidence,
                  })
                  continue

                class_total[true_class_name] += 1
                total_images += 1
                is_correct = pred_class_name == true_class_name
                if is_correct:
                  correct_predictions += 1
                  class_correct[true_class_name] += 1

                # 일반 정답 비교 데이터 상세 내역 저장
                detailed_results.append({
                    "path": img_path,
                    "true_label": true_class_name,
                    "prediction": pred_class_name,
                    "confidence": confidence,
                    "is_correct": is_correct,
                })

          # 메인 요약 지표 출력
          st.success("🎉 일괄 평가 및 분석 완료!")

          if total_images > 0:
            overall_accuracy = (correct_predictions / total_images) * 100
            st.metric(
                label="전체 정확도 (Overall Accuracy)",
                value=f"{overall_accuracy:.2f}%",
                delta=f"총 {total_images}장 평가 완료",
            )

            # 클래스별 요약 테이블
            st.markdown("### 📈 클래스별 상세 정확도 요약")
            acc_data = []
            for cls in valid_classes:
              if cls != "noLabel" and class_total[cls] > 0:
                cls_acc = (class_correct[cls] / class_total[cls]) * 100
                acc_data.append({
                    "클래스명": cls,
                    "정확도(%)": f"{cls_acc:.2f}%",
                    "맞춘 개수": f"{class_correct[cls]} / {class_total[cls]}장",
                })
            st.table(acc_data)

            # 이미지별 판별 내역 리스트 출력 (필터링 기능 추가)
            st.markdown("---")
            st.markdown("### 🖼️ 이미지별 개별 판별 결과 열람")

            # filter_option = st.radio(
            #     "필터 선택",
            #     ["전체 보기", "정답 맞춘 항목만", "틀린 항목만"],
            #     horizontal=True,
            # )

            filtered_results = detailed_results
            # if filter_option == "정답 맞춘 항목만":
            #   filtered_results = [
            #       item for item in detailed_results if item["is_correct"]
            #   ]
            # elif filter_option == "틀린 항목만":
            #   filtered_results = [
            #       item for item in detailed_results if not item["is_correct"]
            #   ]

            st.write(f"총 **{len(filtered_results)}장**의 결과가 표시됩니다.")

            # 결과를 카드형태로 그리드 출력
            for item in filtered_results:
              img_filename = os.path.basename(item["path"])
              status_icon = "✅ 정답" if item["is_correct"] else "❌ 오답"

              with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                  try:
                    img = Image.open(item["path"])
                    st.image(img, width=120)
                  except Exception:
                    st.write("이미지 로드 불가")

                with cols[1]:
                  st.markdown(f"**파일명:** `{img_filename}`")
                  st.markdown(
                      f"**상태:** {status_icon} | **정답 폴더:**"
                      f" `{item['true_label']}`"
                  )
                  st.markdown(
                      f"**AI 예측:** `{item['prediction']}` (확신도:"
                      f" **{item['confidence']:.2f}%**)"
                  )
                st.divider()

          # noLabel 결과 출력
          if nolabel_results:
            st.markdown("---")
            st.markdown("### 🏷️ 'noLabel' 폴더 이미지 예측 결과")
            for item in nolabel_results:
              file_name = os.path.basename(item["path"])
              cols = st.columns([1, 3])
              with cols[0]:
                try:
                  img = Image.open(item["path"])
                  st.image(img, width=120)
                except Exception:
                  pass
              with cols[1]:
                st.markdown(f"**파일명:** `{file_name}`")
                st.markdown(
                    f"🌿 **예측 결과:** `{item['prediction']}` | 🔍 **확신도:**"
                    f" `{item['confidence']:.2f}%`"
                )
              st.divider()

        except Exception as e:
<<<<<<< HEAD
<<<<<<< HEAD
          st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
=======
          st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
>>>>>>> ffb38c92000ce000a2ba35f5a488c6cb06052060
=======
          st.error(f"평가 중 오류가 발생했습니다: {str(e)}")
>>>>>>> ffb38c92000ce000a2ba35f5a488c6cb06052060
