import cv2
import numpy as np
import easyocr
import re

EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
OCR_TH = 0.2


def detectx(frame, model):
    results = model(frame)
    detections = results[0].boxes
    boxes = detections.xyxy.cpu().numpy()
    labels = detections.cls.cpu().numpy()
    confidences = detections.conf.cpu().numpy()
    return labels, boxes, confidences

def plot_boxes(results, frame, classes, conf_threshold=0.4):
    labels, boxes, confidences = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections before filtering.")

    filtered_indices = confidences >= conf_threshold  # Filter by confidence

    filtered_boxes = boxes[filtered_indices]
    filtered_labels = labels[filtered_indices]
    filtered_confidences = confidences[filtered_indices]
    n_filtered = len(filtered_labels)

    print(f"[INFO] Total {n_filtered} detections after filtering (confidence threshold = {conf_threshold}).")
    print(f"[INFO] Looping through filtered detections...")

    for i in range(n_filtered):
        row = filtered_boxes[i]

        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        text_d = classes[int(filtered_labels[i])]
        coords = [x1, y1, x2, y2]
        plate_num = recognize_plate_easyocr(
            img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"{plate_num}({filtered_confidences[i]:.2f})",  # Show confidence
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return frame

def recognize_plate_easyocr(img, coords, reader, region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]


    # Preprocessing
    gray = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)

    ocr_result = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-', detail=0)

    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)

    if text:  # If EasyOCR detected any text
        return " ".join(text)
    else:
        return "Unrecognized"


def filter_text(region, ocr_result, region_threshold):
    plate = []
    print(ocr_result)
    for text in ocr_result:
        plate.append(text)
    return plate