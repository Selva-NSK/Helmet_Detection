from ultralytics import YOLO
import os
import cv2
import shutil
from datetime import datetime
from tqdm.auto import tqdm
import subprocess

model = YOLO('best.pt',verbose=False)
model2 = YOLO('license_plate_detector.pt', verbose=False)

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(img, classes=[], conf=0.5):
    results = predict(model, img, classes, conf=conf)
    res = []
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] != 'motorcyclist':
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                res.append(result.names[int(box.cls[0])])

    results = predict(model2, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] != 'motorcyclist':
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                res.append(result.names[int(box.cls[0])])

    return img, res

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, float(fps), (frame_width, frame_height))

    return writer

def get_vid_predictions(vid_path):
    result_score = []
    folder_path = os.path.join('results', datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
    os.makedirs(folder_path)

    shutil.copy(vid_path, folder_path)
    output_filename = os.path.join(folder_path, "output.mp4")

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = create_video_writer(cap, output_filename)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            success, vid_frame = cap.read()

            if not success:
                break

            result_img,_ = predict_and_detect(vid_frame, classes=[], conf=0.2)
            if _:
                result_score = _
                cv2.imwrite('output_image.jpg', result_img)
            writer.write(result_img)
            pbar.update(1)

        cap.release()  # Release the video capture
        writer.release()  # Release the video writer

    temp_file_result = output_filename
    convertedVideo = os.path.join(folder_path,'output2.mp4')
    subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))

    return os.path.abspath(convertedVideo),result_score

if __name__ == '__main__':
    vid_path = 'sample.mp4'
    path = get_vid_predictions(vid_path)
    print(path)
