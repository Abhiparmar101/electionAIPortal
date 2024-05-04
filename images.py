import subprocess
import os
from datetime import datetime
from multiprocessing import Pool
import time
import csv

def capture_frame(stream_url):
    output_dir = "captured_images"
    log_file = "error_log.txt"
    retry_count = 3

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        identifier = stream_url.split("/")[-1]
        output_path = os.path.join(output_dir, f"{identifier}_{timestamp}.jpg")
        command = ["ffmpeg", "-i", stream_url, "-frames:v", "1", output_path, "-y"]
        error_message = ""

        for attempt in range(retry_count):
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if process.returncode == 0:
                print(f"Image captured and saved as {output_path}")
                break
            else:
                error_message = stderr.decode('utf-8')
                if "Server returned 404 Not Found" in error_message or "Connection refused" in error_message:
                    print(f"Stream not available: {error_message}")
                    log_capture_error(log_file, stream_url, timestamp, "Stream not available or closed.")
                    return  # Exit if the stream is confirmed to be closed or not available

        if process.returncode != 0:
            log_capture_error(log_file, stream_url, timestamp, error_message)
            print(f"Failed after {retry_count} retries. Logged the error.")

        time.sleep(60)  # Wait for 1 minute before next capture attempt

def log_capture_error(log_file, stream_url, timestamp, error_message):
    with open(log_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, stream_url, error_message])

def main(stream_urls):
    if not os.path.exists("error_log.csv"):
        with open("error_log.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Stream URL", "Error Message"])

    with Pool(len(stream_urls)) as pool:
        pool.map(capture_frame, stream_urls)

if __name__ == "__main__":
    with open("newrtmp.txt", "r") as file:
        stream_urls = [line.strip() for line in file if line.strip()]
    main(stream_urls)
