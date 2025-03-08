import cv2
import numpy as np
from datetime import datetime, timedelta
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import csv
import os

SCREEN_WIDTH = 2650
SCREEN_HEIGHT = 1600

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

model = YOLO("best.pt")  # Load the pre-trained YOLO model

classNames = ['bottle', 'cap', 'cap missing', 'damaged plastic', 'label', 'label missing']

COLOR_BOTTLE = (255, 0, 0)
COLOR_PRESENT = (0, 255, 0)
COLOR_DEFECTIVE = (0, 0, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_STATUS_BACKGROUND = (255, 255, 255)

font_path = "Times new roman.ttf"

csv_file = "bottle_data.csv"
screenshots_dir = "screenshots"
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

bottle_counter = 0  # Global counter to track unique bottles
previous_bottle_number = -1  # Track the previous bottle number
last_saved_time = datetime.now() - timedelta(seconds=5)  # Track last save time
bottle_in_frame = False  # Track if a bottle is currently in the frame
bottle_last_seen_time = datetime.now()  # Time when bottle was last seen

# Minimum time a bottle needs to be in the frame to be considered valid
TIME_THRESHOLD_IN_FRAME = 1.5  # 1.5 seconds


# Bounding box center
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


# Bounding box distance
def bbox_distance(bbox1, bbox2):
    x1, y1 = bbox_center(bbox1)
    x2, y2 = bbox_center(bbox2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


DISTANCE_THRESHOLD = 100  # Define the threshold for distance


# Load custom font
def put_custom_text(img, text, position, font_size, font_color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Setup UI layout
def setup_ui(img):
    h, w, _ = img.shape
    status_area_width = int(w * 0.3)
    detection_area_width = int(w * 0.3)
    non_detected_width = (w - detection_area_width - status_area_width) // 2
    cv2.rectangle(img, (0, 0), (non_detected_width, h), -1)
    cv2.rectangle(img, (w - non_detected_width - status_area_width, 0), (w - status_area_width, h), -1)
    cv2.rectangle(img, (non_detected_width, 0), (w - status_area_width - non_detected_width, h), (255, 255, 255), 8)
    cv2.rectangle(img, (w - status_area_width, 0), (w, h), COLOR_STATUS_BACKGROUND, -1)
    return non_detected_width, w - status_area_width - non_detected_width, w - status_area_width


# Display bottle status in the UI
def display_status(img, bottle_status):
    h, w, _ = img.shape
    status_area_start = int(w * 0.7)
    
    overlay = img.copy()
    box_width = int(w * 0.3) - 20
    box_height = 1500
    cv2.rectangle(overlay, (status_area_start + 10, 20), (status_area_start + 10 + box_width, 1420), (230, 230, 230), 5)
    alpha = 7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    status_color = COLOR_PRESENT if bottle_status['Status'] == 'Non-Defective' else COLOR_DEFECTIVE

    img = put_custom_text(img, f"       DETAILS", (status_area_start + 20, 30), 90, COLOR_TEXT)
    img = put_custom_text(img, f"________________________________________________________________________", (status_area_start + 20, 130), 20, COLOR_TEXT)
    img = put_custom_text(img, f"Status: {bottle_status['Status']}", (status_area_start + 20, 200), 50, status_color)
    img = put_custom_text(img, f"Cap: {bottle_status['Cap']}", (status_area_start + 20, 300), 45, COLOR_TEXT)
    img = put_custom_text(img, f"Label: {bottle_status['Label']}", (status_area_start + 20, 400), 45, COLOR_TEXT)
    img = put_custom_text(img, f"Plastic: {bottle_status['Plastic']}", (status_area_start + 20, 500), 45, COLOR_TEXT)
    img = put_custom_text(img, f"Production Day: {bottle_status['Day']}", (status_area_start + 20, 1150), 40, COLOR_TEXT)
    img = put_custom_text(img, f"Production Date: {bottle_status['Date']}", (status_area_start + 20, 1250), 40, COLOR_TEXT)
    img = put_custom_text(img, f"Current Time: {bottle_status['Time']}", (status_area_start + 20, 1350), 40, COLOR_TEXT)

    return img


# Draw a bounding box with label
def draw_box(img, x1, y1, x2, y2, label, color):
    padding = 10
    x1_p, y1_p = max(x1 - padding, 0), max(y1 - padding, 0)
    x2_p, y2_p = x2 + padding, y2 + padding
    cv2.rectangle(img, (x1_p, y1_p), (x2_p, y2_p), color, 6)
    img = put_custom_text(img, label, (x1_p, y1_p - 40), 40, color)
    return img


# Get default bottle status
def get_default_bottle_status():
    global bottle_counter
    bottle_counter += 1
    return {
        "Bottle Number": bottle_counter,
        "Cap": "Not Detected",
        "Label": "Not Detected",
        "Plastic": "Good",
        "Status": "--",
        "Day": datetime.now().strftime("%A"),
        "Date": datetime.now().strftime("%d/%m/%y"),
        "Time": datetime.now().strftime("%H:%M:%S"),
    }


# Save bottle data to CSV
def save_to_csv(bottle_status):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=bottle_status.keys())
        if not file_exists:
            writer.writeheader()  # Write the header if file is new
        writer.writerow(bottle_status)


# Save a screenshot of the bottle
def save_screenshot(img, bottle_number):
    screenshot_path = os.path.join(screenshots_dir, f"bottle_{bottle_number}.png")
    cv2.imwrite(screenshot_path, img)


previous_bottle = None
current_bottle_status = get_default_bottle_status()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    current_bottle_status["Day"] = datetime.now().strftime("%A")
    current_bottle_status["Date"] = datetime.now().strftime("%d/%m/%y")
    current_bottle_status["Time"] = datetime.now().strftime("%H:%M:%S")

    track_start, track_end, status_start = setup_ui(img)
    results = model(img)

    bottle_detected = False
    temp_cap = "Not Detected"
    temp_label = "Not Detected"
    temp_plastic = "Good"
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = classNames[int(box.cls[0])]
            
            if x1 >= track_start and x2 <= track_end:
                if label == 'bottle':
                    bottle_detected = True

                    # Track if bottle has been in the frame long enough before saving
                    if not bottle_in_frame:
                        bottle_in_frame = True
                        bottle_last_seen_time = datetime.now()
                    elif (datetime.now() - bottle_last_seen_time).total_seconds() >= TIME_THRESHOLD_IN_FRAME:
                        # Save only when the bottle is in frame long enough
                        if previous_bottle is None or bbox_distance(previous_bottle, [x1, y1, x2, y2]) > DISTANCE_THRESHOLD:
                            current_bottle_status = get_default_bottle_status()
                            previous_bottle = [x1, y1, x2, y2]
                    img = draw_box(img, x1, y1, x2, y2, "Bottle", COLOR_BOTTLE)

                elif label == 'cap':
                    temp_cap = "Detected"
                    img = draw_box(img, x1, y1, x2, y2, "Cap", COLOR_PRESENT)
                elif label == 'cap missing':
                    temp_cap = "Missing"
                    img = draw_box(img, x1, y1, x2, y2, "Cap Missing", COLOR_DEFECTIVE)
                elif label == 'label':
                    temp_label = "Detected"
                    img = draw_box(img, x1, y1, x2, y2, "Label", COLOR_PRESENT)
                elif label == 'label missing':
                    temp_label = "Missing"
                    img = draw_box(img, x1, y1, x2, y2, "Label Missing", COLOR_DEFECTIVE)
                elif label == 'damaged plastic':
                    temp_plastic = "Damaged"
                    img = draw_box(img, x1, y1, x2, y2, "Damaged Plastic", COLOR_DEFECTIVE)

    if bottle_detected:
        current_bottle_status["Cap"] = temp_cap
        current_bottle_status["Label"] = temp_label
        current_bottle_status["Plastic"] = temp_plastic
        if temp_cap == "Missing" or temp_label == "Missing" or temp_plastic == "Damaged":
            current_bottle_status["Status"] = "Defective"
        else:
            current_bottle_status["Status"] = "Non-Defective"

        current_time = datetime.now()
        # Save CSV and Screenshot if it's a new bottle and sufficient time has passed
        if current_bottle_status["Bottle Number"] != previous_bottle_number and (current_time - last_saved_time).seconds >= 2:
            save_to_csv(current_bottle_status)
            save_screenshot(img, current_bottle_status["Bottle Number"])
            previous_bottle_number = current_bottle_status["Bottle Number"]
            last_saved_time = current_time  # Update last saved time

    img = display_status(img, current_bottle_status)
    cv2.imshow("Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
