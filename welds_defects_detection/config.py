import platform

IS_INTEL = "intel" in platform.processor().lower() or "i386" in platform.processor().lower()

MODEL_SIZES = {
    "n": "Быстрее",
    "s": "Быстрее",
    "m": "Точнее",
    "l": "Точнее",
}

MODELS_OV = {
    "m": "models/20240615_5_classes_640_8m_100_aug/weights/best_openvino_model",
    "l": "models/20240615_5_classes_640_8l_150_90_10_iter_train_aug/weights/best_openvino_model",
    "n": "models/20240615_5_classes_640_8n_150_90_10_iter_train_aug/weights/best_openvino_model",
    "s": "models/20240615_5_classes_640_8s_150_90_10_iter_train_aug/weights/best_openvino_model",
}
MODELS = {
    "m": "models/20240615_5_classes_640_8m_100_aug/weights/best.pt",
    "l": "models/20240615_5_classes_640_8l_150_90_10_iter_train_aug/weights/best.pt",
    "n": "models/20240615_5_classes_640_8n_150_90_10_iter_train_aug/weights/best.pt",
    "s": "models/20240615_5_classes_640_8s_150_90_10_iter_train_aug/weights/best.pt",
}
