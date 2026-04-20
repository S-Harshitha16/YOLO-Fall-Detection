import cv2
import time
import os
import csv
import threading
import pyttsx3
from ultralytics import YOLO
from twilio.rest import Client

# ================== VOICE ==================
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ================== TWILIO CONFIG ==================
account_sid = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
client = Client(account_sid, auth_token)

FROM_WHATSAPP = "whatsapp:+14155238886"
CALL_FROM = "+14635001360"
SMS_FROM = "+14635001360"

TO_WHATSAPP_LIST = ["whatsapp:+91xxxxxxxxxx","whatsapp:+91xxxxxxxxxx"]
CALL_TO_LIST = ["+91xxxxxxxxxx","+91xxxxxxxxxx"]
SMS_TO_LIST = ["+91xxxxxxxxxx","+91xxxxxxxxxx"]

PUBLIC_URL = "https://unwinning-scruffily-hannelore.ngrok-free.dev"

# ================== PARAMETERS ==================
NO_MOVEMENT_TIME = 3
ALERT_GAP = 60

fall_start_time = None
last_alert_time = 0
alert_running = False
voice_given = False

# ================== MODEL ==================
model = YOLO("best.pt")  # your trained YOLO model

# ================== LOG FILE ==================
if not os.path.exists("fall_log.csv"):
    with open("fall_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Image"])

def log_event(image_path):
    full_time = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("fall_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([full_time, image_path])
    except:
        print("Close CSV file if open!")

# ================== ALERT ==================
def send_alert(image_path):
    global alert_running, last_alert_time
    alert_running = True
    last_alert_time = time.time()
    try:
        image_url = f"{PUBLIC_URL}/{image_path}"
        location_link = "https://share.google/VNAee9AUSUtva2426"
        for i in range(len(TO_WHATSAPP_LIST)):
            # WhatsApp
            client.messages.create(
                body=f"🚨 Emergency Alert!\nFall detected.\nLocation: {location_link}",
                from_=FROM_WHATSAPP,
                to=TO_WHATSAPP_LIST[i],
                media_url=[image_url]
            )
            # Call
            client.calls.create(
                to=CALL_TO_LIST[i],
                from_=CALL_FROM,
                twiml="<Response><Say>Emergency! Fall detected.</Say></Response>"
            )
            # SMS
            client.messages.create(
                body=f"🚨 Fall detected! Location: {location_link}",
                from_=SMS_FROM,
                to=SMS_TO_LIST[i]
            )
        print("✅ Alerts Sent")
    except Exception as e:
        print("Alert Error:", e)
    alert_running = False

def trigger_alert(image_path):
    threading.Thread(target=send_alert, args=(image_path,), daemon=True).start()

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit()
print("System Started 🚀")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,(640,480))

    # Time overlay
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame,current_time,(10,460),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # YOLO prediction
    results = model(frame, conf=0.4, imgsz=320)
    fall_detected = False
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label.lower() == "fall":
                fall_detected = True
        frame = r.plot()

    # Status overlay
    status = "Monitoring"
    color = (0,255,0)
    if fall_detected:
        status = "Fall Detected"
        color = (0,255,255)
    if alert_running:
        status = "Alert Sent"
        color = (0,0,255)
    cv2.putText(frame,f"Status: {status}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    # FALL LOGIC
    if fall_detected:
        if fall_start_time is None:
            fall_start_time = time.time()
        elapsed = time.time() - fall_start_time
        if elapsed > NO_MOVEMENT_TIME:
            cv2.putText(frame,"CONFIRMED FALL!",(20,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            if not voice_given:
                speak("Are you okay?")
                voice_given = True
                time.sleep(10)
            if (not alert_running and time.time()-last_alert_time > ALERT_GAP):
                image_name = f"fall_{int(time.time())}.jpg"
                cv2.imwrite(image_name, frame)
                log_event(image_name)
                time.sleep(2)
                print("🚨 Sending Alert...")
                trigger_alert(image_name)
                fall_start_time = None
    else:
        fall_start_time = None
        voice_given = False

    cv2.imshow("Smart Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
