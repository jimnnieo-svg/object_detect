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
