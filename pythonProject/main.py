# # Codigo Para Criar O Modelo
# from ultralytics import YOLO
#
# # Load YOLOv10n model from scratch
# model = YOLO("yolov10n.yaml")
#
# # train the model
# model.train(data="data.yaml", epochs=2)

# from ultralytics import YOLO
# import cv2
#
# model = YOLO("runs/detect/train10/weights/best.pt")
#
# # Load an image
# image_path = 'image.jpg'
# image = cv2.imread(image_path)
#
# image_path = "image.jpg"
# output_dir = "."
# results = model.predict(source=image_path, save=True, project=output_dir)
# print(results)

from ultralytics import YOLO
import cv2
import functions

model = YOLO("runs/detect/train10/weights/best.pt")
classes = model.names

image_path = 'image_2.jpg'

frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

labels, boxes, confidences = functions.detectx(frame, model=model)

frame = functions.plot_boxes((labels, boxes, confidences), frame, classes)

output_path = "output_image.jpg"
cv2.imwrite(output_path, frame)




video_path = "video.mp4"
video_output_path = "output_video.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
out = cv2.VideoWriter(video_output_path, codec, fps, (width, height))

frame_no = 1

cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    if ret and frame_no % 1 == 0:
        print(f"[INFO] Working with frame {frame_no} ")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = functions.detectx(frame, model=model)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = functions.plot_boxes(results, frame, classes=classes)

        cv2.imshow("vid_out", frame)
        if video_output_path:
            print(f"[INFO] Saving output video. . . ")
            out.write(frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        frame_no += 1

print(f"[INFO] Clening up. . . ")
### releaseing the writer
out.release()

## closing all windows
cv2.destroyAllWindows()
