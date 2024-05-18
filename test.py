from ultralytics import YOLO
import os
import cv2
import csv

def detect_on_video(video_path, hand_save_dir, coord_save_path) -> None:
    video_path_out = '{}_prediction.mp4'.format(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'best.pt')
    model = YOLO(model_path)
    threshold = 0.7

    frame_count = 0
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            region = frame[int(y1):int(y2), int(x1):int(x2)]
            if score > threshold and int(class_id) == 0:
                save_hand_and_coordinates(region, x1, y1, x2, y2, hand_save_dir, coord_save_path, video_path, frame_count)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        out.write(frame)
        ret, frame = cap.read()
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_on_webcam(hand_save_dir, coord_save_path) -> None:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    model_path = os.path.join('.', 'runs', 'detect', 'train4', 'weights', 'best.pt')
    model = YOLO(model_path)
    threshold = 0.8

    frame_count = 0
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            region = frame[int(y1):int(y2), int(x1):int(x2)]
            if score > threshold and int(class_id) == 0:
                save_hand_and_coordinates(region, x1, y1, x2, y2, hand_save_dir, coord_save_path, 'webcam_frame', frame_count)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            ret, frame = cap.read()
            frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def save_hand_and_coordinates(region, x1, y1, x2, y2, hand_save_dir, coord_save_path, original_name, frame_count=None):
    if not os.path.exists(hand_save_dir):
        os.makedirs(hand_save_dir)
    if not os.path.exists(coord_save_path):
        os.makedirs(coord_save_path)

    base_name = os.path.basename(original_name)
    name, ext = os.path.splitext(base_name)

    if frame_count is not None:
        img_name = f'{name}_frame{frame_count}.jpg'
    else:
        img_name = f'{name}.jpg'

    cv2.imwrite(os.path.join(hand_save_dir, img_name), region)

    coord_file = os.path.join(coord_save_path, 'coordinates.csv')
    with open(coord_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, x1, y1, x2, y2])
def detect_on_img(img_path, hand_save_dir, coord_save_path) -> None:
    img_path_out = '{}_prediction.png'.format(img_path)
    model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')
    model = YOLO(model_path)
    threshold = 0.5
    img = cv2.imread(img_path)
    results = model(img)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        region = img[int(y1):int(y2), int(x1):int(x2)]
        if score > threshold and int(class_id) == 0:
            save_hand_and_coordinates(region, x1, y1, x2, y2, hand_save_dir, coord_save_path, img_path)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1, cv2.LINE_AA)

    img = cv2.resize(img, (720, 480))
    while True:
        cv2.imshow('out', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(img_path_out, img)
    cv2.destroyAllWindows()
def main():
    VIDEOS_DIR = os.path.join('.', 'videos')
    HAND_SAVE_DIR = os.path.join('.', 'results', 'hand_detections')
    COORD_SAVE_PATH = os.path.join('.', 'results', 'coordinates')

    img_path = ('images/23.jpg')
    video_path = os.path.join(VIDEOS_DIR, 'video_name.mp4')

    detect_on_img(img_path, HAND_SAVE_DIR, COORD_SAVE_PATH)

if __name__ == "__main__":
    main()
