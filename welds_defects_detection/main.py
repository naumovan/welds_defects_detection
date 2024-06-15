import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sahi import AutoDetectionModel

from welds_defects_detection.SahiOpenVino import load_openvino_sahi
from welds_defects_detection.config import MODELS, IS_INTEL, MODEL_SIZES
from welds_defects_detection.detection import detect_defects


@st.cache_resource()
def get_model(size) -> tuple[YOLO, list[str]]:
    model = YOLO(MODELS[size])
    classes = model.names
    return model, classes


@st.cache_resource()
def get_model_sahi(size) -> tuple[YOLO, list[str]]:
    if IS_INTEL:
        model = load_openvino_sahi(
            model_path=MODELS[size],
            confidence_threshold=0,
            device="cpu",  # or 'cuda:0'
        )
    else:
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=MODELS[size],
            confidence_threshold=0,
            device="cpu",  # or 'cuda:0'
        )

    _, classes = get_model(size)
    return model, classes


def load_image():
    image_file = st.file_uploader("Загрузить изображение", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        bytes_data = image_file.read()
        image_orig = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        return image_orig
    return None


def frontend():
    st.set_page_config(
        page_title="Детекция сварных швов",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Детекция сварных швов")
    st.sidebar.title("Настройки детекции")

    model_size = st.sidebar.select_slider(
        "Выбрать модель",
        options=MODEL_SIZES.keys(),
        format_func=lambda x: MODEL_SIZES[x],
    )

    use_sahi = st.sidebar.checkbox("Детекция мелких объектов")

    with st.sidebar.expander("Дополнительные настройки", expanded=False):
        # Настройка порога уверенности
        confidence = float(st.slider("Фильтр уверенности", 25, 100, 40)) / 100
        # Подготовка модели и настройка SAHI
        if use_sahi:
            st.title("Настройка SAHI")
            size_window_sahi = st.slider("Размер скользящего окна", 64, 512, 256, 64)
        else:
            size_window_sahi = None

    image = load_image()
    if image is not None:
        with st.spinner("Идет обработка изображения..."):
            if not use_sahi:
                model, model_classes = get_model(model_size)
            else:
                model, model_classes = get_model_sahi(model_size)
                model.confidence_threshold = confidence
            class_counts, res_image = detect_defects(
                image, model, model_classes, use_sahi, confidence,
                size_window_sahi,
            )
            st.image(res_image, caption="Результат детекции", use_column_width=True)

            if class_counts:
                st.sidebar.title("Диаграмма обнаруженных классов")
                fig, ax = plt.subplots()
                ax.bar(class_counts.keys(), class_counts.values())
                ax.set_xlabel('Классы')
                ax.set_ylabel('Количество')
                st.sidebar.pyplot(fig)


if __name__ == "__main__":
    frontend()
