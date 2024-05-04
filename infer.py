from ultralytics import YOLO
import os
import cv2
from pathlib import Path
import time
import requests
from datetime import datetime
from create_table import init_db
import sqlite3

# Initialize YOLO model
model = YOLO('yolov8x.pt')  # Ensure this path to your model is correct

# Directory paths
input_folder = 'captured_images'  # Folder with input images
output_folder = 'blobdrive/t'  # Folder to save processed images
processed_files = set()  # To keep track of processed files
init_db()
def send_data_to_server(data):
    url = "https://ai-analytics-election-igrgh.ondigitalocean.app/api/post-analytics"
    try:
        response = requests.post(url, json=data)
        print(f"Data sent to server: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending data to server: {e}")



def insert_metadata_into_db(data):
    # Connect to SQLite3 database
    conn = sqlite3.connect('metadata.db')
    c = conn.cursor()

    # Insert metadata into the table
    c.execute('''INSERT INTO metadata 
                 (cameraid, timestamp, imageurl,personcount) 
                 VALUES (?, ?, ?, ?)''',
              (data["cameraid"], data["timestamp"], data["imageurl"],
               data["personcount"]))
    conn.commit()
    conn.close()

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:  # Continuously check for new images
        file_list = Path(input_dir).glob('*.jpg')
        for image_path in file_list:
            if image_path.name not in processed_files:
                # Predict using YOLOv8 model
                results = model.predict(source=str(image_path), save=False)

                img = cv2.imread(str(image_path))
                total_person_count = 0
                # Draw bounding boxes on the image
                for result in results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        if cls == 0:  # Assuming class 0 is 'person'
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            total_person_count += 1

                # Put text for total person count on the image
                # cv2.putText(img, f'Person Count: {total_person_count}', (350, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                text = f'Person Count: {total_person_count}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(img, (350, 15 - text_size[1]), (400 + text_size[0], 40), (0, 0, 0), -1)
                cv2.putText(img, text, (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Save the processed image
                save_path = os.path.join(output_dir, image_path.name)
                cv2.imwrite(save_path, img)
                print(f"Processed image saved to {save_path}")
                camera_id = image_path.stem.split('_')[0]
                imgurl = "https://inferenceimage.blob.core.windows.net/inferenceimages/t/" + image_path.stem + ".jpg"

                # Send data to server only if person count is greater than or equal to 1
                if total_person_count >= 1:
                    data = {
                        "cameraid": camera_id,
                        "sendtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "imgurl": imgurl,
                        "modelname": "Crowd",
                        "ImgCount": total_person_count,
                        "customerid": "1234567",
                        "streamname": camera_id
                    }
                    data_db = {
                        "cameraid": camera_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "imageurl": imgurl,
                       
                        "personcount": total_person_count,
                       
                    }
                    send_data_to_server(data)
                    # Insert metadata into SQLite database
                    insert_metadata_into_db(data_db)
                    print(data)
                # Add to processed files list
                processed_files.add(image_path.name)

                # Delete the original image file
                try:
                    os.remove(str(image_path))
                    print(f"Original image deleted: {image_path.name}")
                except OSError as e:
                    print(f"Error deleting file {image_path.name}: {e}")

        time.sleep(10)  # Check for new images every 10 seconds

if __name__ == '__main__':
    process_images(input_folder, output_folder)
