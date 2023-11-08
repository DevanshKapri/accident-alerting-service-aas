import os
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

from dotenv import load_dotenv

load_dotenv()

model = load_model('model/AccidentDetectionModel.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.secret_key = os.getenv('SECRET_KEY')

# Function to check if the filename extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_frame(img):
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction=(model.predict(img_batch) > 0.5).astype("int32")
    if(prediction[0][0]==0):
        return("Accident Detected")
    else:
        return("No Accident")

# Function to process the video file
def process_video(video_path):
    image=[]
    img_height = 300
    img_width = 300

    c=1
    cap= cv2.VideoCapture(video_path)
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            cap.release()
            cap= cv2.VideoCapture(video_path)
            grabbed, frame = cap.read()
        if c%30==0:
            if not grabbed:
                print("Error in capturing frame...retrying...")
            print(c)
            resized_frame=tf.keras.preprocessing.image.smart_resize(frame, (img_height, img_width), interpolation='bilinear')
            result = predict_frame(resized_frame)
            print(result)
            if (result == "Accident Detected"):
                image.append(frame)
            if(len(image)==2):
                break
        c+=1

        cap.release()
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    alert = False
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            frames_with_accident_base64 = []

            frames_with_accident = process_video(file_path)
            for frame in frames_with_accident:
                # Convert the frame to a base64-encoded image
                print(type(frame))
                _, buffer = cv2.imencode(".jpg", frame)
                frame_base64 = base64.b64encode(buffer).decode()
                frames_with_accident_base64.append(frame_base64)

                # Set alert to True if an accident is detected
                if not alert:
                    alert = True
                    alert_file = filename.split('.')[0]

            return render_template('index.html', frames_with_accident_base64=frames_with_accident_base64, alert=alert, alert_file=alert_file)

    return render_template('index.html')

@app.route('/approve_alert', methods=['POST'])
def approve_alert():
    alert_file = request.form.get('alert_file')
    if alert_file:
        # Send an alert using the backend with the name of the file
        send_alert(alert_file)
    flash('Alert has been sent to Emergency Services!')
    return redirect('/')

from twilio.rest import Client

def send_alert(file_name):
    print("Sending alert for location {}".format(file_name))
    

    account_sid = 'ACf0e77f236d212680b6eb1899f0ea8c3b'
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='Accident Alert, SOS, Location: {}'.format(file_name),
    to='whatsapp:+917302489729'
    )

    print(message.sid)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
