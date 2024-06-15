import cv2
import numpy as np
from sahi.predict import get_sliced_prediction


def detect_defects(image, model, model_classes, use_sahi, confidence, size_window_sahi=None):
    class_counts = {}

    if not use_sahi:
        # Используем YOLO для детектирования
        res = model(image, conf=confidence)
        boxes = res[0].boxes
        res_image = res[0].plot()

        for box in boxes:
            cls = box.cls
            class_name = model.names[int(cls)]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        return class_counts, res_image

    # Используем SAHI для детектирования с настройками
    results = get_sliced_prediction(
        image,
        model,
        slice_height=size_window_sahi,
        slice_width=size_window_sahi,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    if not results.object_prediction_list:
        return class_counts, image

    boxes_xy, boxes_scores, boxes_classes = [], [], []
    for result in results.object_prediction_list:
        boxes_xy.append(result.bbox.to_xyxy())
        boxes_scores.append(result.score.value)
        boxes_classes.append(result.category.id)

        class_name = model_classes[result.category.id]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    boxes_xy = np.array(boxes_xy, dtype=int)
    boxes_scores = np.array(boxes_scores)
    boxes_classes = np.array(boxes_classes, dtype=int)

    res_image = image.copy()

    for box, score, cls in zip(boxes_xy, boxes_scores, boxes_classes):
        if len(box) == 4:  # Пропускаем, если нет обнаруженных объектов
            x1, y1, x2, y2 = box
            cv2.rectangle(res_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(res_image, f'{model_classes[cls]} {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return class_counts, res_image
