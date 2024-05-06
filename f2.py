import threading
import logging
import csv
import cv2
import os
import subprocess as sp
import requests
from flask import Flask, jsonify
from datetime import datetime, timedelta
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

app = Flask(__name__)

# Load and configure the YOLO model
model = YOLO('yolov8s.pt').to(device)
confidence_threshold = 0.4
save_path = "/home/torqueai/workspace/electionDemo/blobdrive/a"
os.makedirs(save_path, exist_ok=True)

# Setup for logging and CSV recording
logging.basicConfig(level=logging.INFO)
csv_file = open('server_responses2.csv', 'w', newline='', buffering=1)
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Image Filename', 'Total Person Count', 'Stream Name', 'Status Code', 'Response Text'])

stream_threads = {}
stream_watchdogs = {}

def read_camera_urls(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def send_metadata_to_server(image_filename, total_person_count, stream_name):
    img_url = f"https://inferenceimage.blob.core.windows.net/inferenceimages/a/{image_filename}"
    url = "https://ai-analytics-election-igrgh.ondigitalocean.app/api/post-analytics"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
    data = {
        "cameraid": stream_name,
        "sendtime": timestamp,
        "imgurl": img_url,
        "modelname": "Crowd",
        "ImgCount": total_person_count,
        "customerid": "1234567",
        "streamname": "Goa"
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        logging.info(f"Data sent successfully! Response: {response.text}")
        response_text = response.text
    except requests.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        response_text = 'Failed to send data'
        response = None
    finally:
        csv_writer.writerow([timestamp, image_filename, total_person_count, stream_name, response.status_code if response else 'Failed', response_text])
        return response

class StreamWatchdog:
    def __init__(self, timeout, restart_function, args, csv_writer):
        self.timeout = timeout
        self.restart_function = restart_function
        self.args = args
        self.last_update = datetime.now()
        self.running = False
        self.csv_writer = csv_writer
        self.thread = threading.Thread(target=self.watch)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def update(self):
        self.last_update = datetime.now()

    def watch(self):
        while self.running:
            if datetime.now() - self.last_update > timedelta(seconds=self.timeout):
                logging.warning(f"Watchdog timeout reached. Restarting stream: {self.args}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
                stream_name = self.args[1]
                self.csv_writer.writerow([timestamp, 'NA', 'NA', stream_name, 'Watchdog Timeout', 'Restarting stream'])
                self.restart_function(*self.args)
            threading.Event().wait(1)

def restart_stream(camera_url, stream_key):
    if stream_key in stream_threads:
        stream_threads[stream_key].join()
    thread = threading.Thread(target=process_and_stream_frames, args=(camera_url, stream_key))
    thread.start()
    stream_threads[stream_key] = thread
    stream_watchdogs[stream_key].update()

def process_and_stream_frames(camera_url, stream_key):
    cap = cv2.VideoCapture(camera_url)
    stream_name = camera_url.split('/')[-1].split('.')[0]
    modified_camera_url = f"rtmp://mgele6.vmukti.com:80/live-record/{stream_name}_ai"
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '{}x{}'.format(width, height),
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'veryfast',
        '-f', 'flv',
        modified_camera_url
    ]
    process = sp.Popen(command, stdin=sp.PIPE)
    last_capture_time = datetime.now() - timedelta(minutes=1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results[0]
            people_boxes = [box for box in detections.boxes.data if box[-1] == 0 and box[4] > confidence_threshold]
            current_time = datetime.now()

            # Draw bounding boxes and labels on the frame for inference
            for box in people_boxes:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                label = f"Person"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Check if a minute has passed since the last capture and if there are people detected
            if people_boxes and (current_time - last_capture_time) >= timedelta(minutes=1):
                last_capture_time = current_time
                image_filename = f"{stream_name}_{current_time.strftime('%Y%m%d%H%M%S')}.jpg"
                image_path = os.path.join(save_path, image_filename)
                cv2.imwrite(image_path, frame)  # Save the modified frame as an image
                send_metadata_to_server(image_filename, len(people_boxes), stream_name)  # Send metadata
                print(f"Captured and sent data for {len(people_boxes)} person(s) at {current_time.strftime('%Y-%m-%d %H:%M:%S')}.")

            process.stdin.write(frame.tobytes())
            stream_watchdogs[stream_key].update()

    finally:
        cap.release()
        process.terminate()
        process.wait()
        logging.info("Stream and process ended.")
        restart_stream(camera_url, stream_key)

@app.route('/set_model', methods=['POST'])
def set_model_and_stream():
    model_name = "crowd"
    camera_urls = read_camera_urls('num_streams2.txt')
    for idx, camera_url in enumerate(camera_urls):
        stream_key = f"{camera_url}_{model_name}_{idx}"
        if stream_key in stream_watchdogs:
            stream_watchdogs[stream_key].stop()
        watchdog = StreamWatchdog(10, restart_stream, (camera_url, stream_key), csv_writer)
        watchdog.start()
        stream_watchdogs[stream_key] = watchdog
        restart_stream(camera_url, stream_key)
        logging.info(f"Streaming started for model_name: {model_name}, camera_url: {camera_url}")
    return jsonify({'message': 'Streaming started for all cameras'})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
