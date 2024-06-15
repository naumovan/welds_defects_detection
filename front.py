# Python In-built packages
import cv2
import numpy as np

import streamlit as st

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.utils.cv import visualize_object_predictions


MODEL_PATH_NANO = "models/20240615_5_classes_640_8n_150_90_10_iter_train_aug/weights/best.pt"

st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Детекция сварых швов")


st.sidebar.text("Подготовка модели")

st.sidebar.title("Подготовка модели")

menu = ["Yolov8n", "Yolov8s", "Yolov8n+SAHI", "Yolov8s+SAHI"]
choice = st.sidebar.selectbox("",menu)

confidence = float(st.sidebar.slider(
    "Кофиденс", 25, 100, 40)) / 100

if choice == "Yolov8n":
    model = YOLO(MODEL_PATH_NANO)
elif choice == "Yolov8s":
    model = YOLO(MODEL_PATH_NANO)
elif choice == "Yolov8n+SAHI":
    model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=MODEL_PATH_NANO,
    confidence_threshold=confidence,
    device="cpu",  # or 'cuda:0'
    )
    model_classes = YOLO(MODEL_PATH_NANO).model.names
elif choice == "Yolov8s+SAHI":
    model = YOLO(MODEL_PATH_NANO)

# menu = ["Изображение","Видео"]
# choice = st.sidebar.selectbox("",menu)


# if choice == "Изображение":
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                    "filesize":image_file.size}
    st.write(file_details)
    bytes_data = image_file.read()
    image_orig = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

    image = image_orig.copy()

    col1, col2 = st.columns(2)

    with col1:

        st.image(image, caption="Загруженное изображение",
                use_column_width=True)


    with col2:
        if st.sidebar.button('Определить дефекты'):
            if choice != "Yolov8n+SAHI":
                res = model(image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res[0].plot(), caption='Результат детекции',
                        use_column_width=True)
            else:
                res = get_prediction(image, model)
                res.export_visuals(export_dir="demo_data/")

                results = get_sliced_prediction(
                    image,
                    model,
                    slice_height=128,
                    slice_width=128,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )

                if results.object_prediction_list:

                    boxes_xy, boxes_scores, boxes_classes = [], [], []
                    for result in results.object_prediction_list:
                        boxes_xy.append(result.bbox.to_xyxy())
                        boxes_scores.append(result.score.value)
                        boxes_classes.append(result.category.id)

                    boxes_xy = np.array(boxes_xy, dtype=int)
                    boxes_scores = np.array(boxes_scores)
                    boxes_classes = np.array(boxes_classes, dtype=int)

                    image_visual = image.copy()

                    for box, score, cls in zip(boxes_xy, boxes_scores, boxes_classes):
                        if len(box) == 4:  # Пропускаем, если нет обнаруженных объектов
                            x1, y1, x2, y2 = box
                            cv2.rectangle(image_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            cv2.putText(image_visual, f'{model_classes[cls]} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

                    st.image(image_visual, caption='Результат детекции SAHI',
                            use_column_width=True)
