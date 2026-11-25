# Object Detection | 사진 한 장으로 확인하는 YOLO 추론

카메라로 찍거나 업로드한 이미지에서 어떤 객체가 몇 개 탐지됐는지 확인하는 Python 웹앱입니다. Streamlit에서 입력·추론·결과 확인을 한 흐름으로 구성했습니다. 사전학습 YOLOv8n을 CPU에서 실행하며, 탐지 상자와 라벨·신뢰도·종류별 개수를 보여줍니다. [프로젝트](https://github.com/jimnnieo-svg/object_detect)

## 담당 범위와 기술

본인 계정의 초기 커밋에서 확인할 수 있는 구현 범위는 Streamlit 입력, Pillow·NumPy 전처리, Ultralytics YOLO 추론, OpenCV 시각화와 pandas 결과 표입니다. 이 저장소에서 확인되는 AI 활용은 사전학습 모델의 서비스 통합입니다. 이 프로젝트에 어떤 코딩 도구를 사용했는지, 각 함수의 작성 주체가 누구인지는 기록으로 확인되지 않습니다. [구현 커밋](https://github.com/jimnnieo-svg/object_detect/commit/0729f3cc5d1c646805ec7332e82b6f8bc13709d6)

## 입력을 결과 화면으로 바꾸는 과정

1. 카메라 또는 파일로 이미지를 받아 RGB로 변환합니다.
2. 큰 입력은 종횡비를 유지해 긴 변을 최대 960픽셀로 줄이고, 모델은 Streamlit 리소스 캐시에 보관합니다. 큰 사진을 CPU로 처리할 때의 입력 부담과 반복 로딩을 줄이기 위한 구현입니다.
3. 사용자가 정한 신뢰도 임곗값으로 추론하고 박스·표·종류별 개수를 표시합니다. 검출된 객체가 없을 때는 안내 메시지를 표시합니다.

[전처리·추론·표시 코드](https://github.com/jimnnieo-svg/object_detect/blob/0729f3cc5d1c646805ec7332e82b6f8bc13709d6/app.py)

## 설치와 실행

Python 가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다. 모델 파일이 로컬에 없으면 최초 실행 시 가중치를 받을 수 있는 환경이 필요합니다.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 실제 화면과 검증 상태

현재 공개 저장소에는 실제 실행 캡처나 자동 테스트 결과가 없습니다. 소스에서 입력·표시 경로를 확인했으며 탐지 정확도와 추론 시간은 미측정입니다. 재현 시 같은 사물 사진을 업로드하고 임곗값 두 개에서 박스와 개수 표가 어떻게 달라지는지 기록할 수 있습니다. [설치 요구사항](https://github.com/jimnnieo-svg/object_detect/blob/0729f3cc5d1c646805ec7332e82b6f8bc13709d6/requirements.txt)

다음 개선은 직접 촬영한 사물 사진으로 입력·출력 화면을 확보하고, 무검출 입력과 큰 입력을 포함한 테스트를 남기는 것입니다. Python과 패키지 버전도 함께 기록하면 같은 조건에서 결과를 비교할 수 있습니다.

---

<details>
<summary>기존 프로젝트 문서와 실행 안내</summary>

# Mobile Object Detection App (Streamlit + YOLOv8)

This project is a Streamlit web app that detects objects from:

- Smartphone camera input (`st.camera_input`)
- Uploaded image files (`st.file_uploader`)

The app uses a pretrained YOLOv8 model (`yolov8n.pt`) and runs inference on CPU.

## Features

- Mobile-friendly camera capture
- Image upload support
- YOLOv8 object detection
- Bounding box visualization
- Label + confidence output
- Object count statistics
- User-friendly Streamlit UI

## System Flow

Image Input -> Model Inference -> Detection -> Visualization -> Output

## Tech Stack

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- Pillow

## Project Structure

- `app.py`
  - `load_model()`
  - `preprocess_image()`
  - `run_detection()`
  - `draw_boxes()`
  - `display_results()`
- `requirements.txt`
- `README.md`

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Performance Notes

- Input images are resized during preprocessing to improve CPU inference speed.
- `@st.cache_resource` is used for model loading to avoid reloading on each run.

## Extension Ideas

- AI explanation module for detected objects
- Dangerous object detection and alerting
- Real-time video detection
- Jetson deployment for edge inference

## Submission Checklist

- [ ] Push code to GitHub repository
- [ ] Verify app runs successfully
- [ ] Include this README with setup and usage instructions

</details>
