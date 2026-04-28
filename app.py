import io
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load YOLOv8 model once and keep it in cache."""
    model = YOLO(model_path)
    return model


def preprocess_image(image: Image.Image, max_size: int = 960) -> np.ndarray:
    """Convert PIL image to RGB NumPy array and resize for faster CPU inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    height, width = img_np.shape[:2]
    scale = min(max_size / max(height, width), 1.0)

    if scale < 1.0:
        new_w = int(width * scale)
        new_h = int(height * scale)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img_np


def run_detection(
    model: YOLO, image_np: np.ndarray, conf_threshold: float = 0.25
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Run YOLO detection on CPU and return parsed detections."""
    results = model.predict(
        source=image_np,
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    detections: List[Dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names.get(cls_id, str(cls_id))

            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )

    return detections, image_np


def draw_boxes(image_np: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    vis_img = image_np.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        caption = f"{label} {conf:.2f}"

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(vis_img, (x1, max(y1 - 24, 0)), (x2, y1), (0, 255, 0), -1)
        cv2.putText(
            vis_img,
            caption,
            (x1 + 4, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis_img


def display_results(result_image: np.ndarray, detections: List[Dict[str, Any]]) -> None:
    """Display detection image, object list, and count statistics."""
    st.subheader("Detection Result")
    st.image(result_image, channels="RGB", use_container_width=True)

    if not detections:
        st.info("No objects detected.")
        return

    records = [
        {
            "Object": det["label"],
            "Confidence": f"{det['confidence']:.2f}",
            "Bounding Box (x1, y1, x2, y2)": det["bbox"],
        }
        for det in detections
    ]
    df = pd.DataFrame(records)

    st.subheader("Detected Objects")
    st.dataframe(df, use_container_width=True)

    counts = Counter(det["label"] for det in detections)
    st.subheader("Object Count Statistics")
    st.json(dict(counts))


def main() -> None:
    st.set_page_config(page_title="Mobile Object Detection (YOLOv8)", layout="wide")
    st.title("Mobile Object Detection App")
    st.write(
        "Capture an image from your smartphone camera or upload an image file, "
        "then run YOLOv8 object detection on CPU."
    )

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Camera Input")
        camera_file = st.camera_input("Take a photo")

    with col2:
        st.subheader("2) Upload Input")
        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

    image = None
    if camera_file is not None:
        image = Image.open(io.BytesIO(camera_file.getvalue()))
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)

    if image is not None:
        st.subheader("Input Preview")
        st.image(image, use_container_width=True)

    run_button = st.button("Run Detection", type="primary", use_container_width=True)

    if run_button:
        if image is None:
            st.warning("Please capture or upload an image first.")
            return

        with st.spinner("Loading model and running detection..."):
            model = load_model("yolov8n.pt")
            image_np = preprocess_image(image)
            detections, preprocessed_image = run_detection(
                model, image_np, conf_threshold=conf_threshold
            )
            result_image = draw_boxes(preprocessed_image, detections)

        display_results(result_image, detections)

    st.markdown("---")
    st.markdown("### Possible Extensions")
    st.markdown("- AI explanation for detected objects")
    st.markdown("- Dangerous object alert mode")
    st.markdown("- Real-time video stream detection")
    st.markdown("- Jetson edge deployment")


if __name__ == "__main__":
    main()
