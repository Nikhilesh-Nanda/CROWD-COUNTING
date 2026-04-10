from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time

# ================== CONFIG ==================
model_path  = "./models/weights.pt"                # change to your model path
input_video = "./vid/vid5.mp4"           # change to your video path
conf_thresh = 0.25

# ================== LOAD MODEL ==================
print("Loading YOLO model...")
model = YOLO(model_path)

app = Flask(__name__)

# ================== VIDEO GENERATOR ==================
def generate_frames():
    cap = cv2.VideoCapture(input_video)
    prev_time = time.time()
    frame_id = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_id += 1

        # Run inference
        results = model(frame, conf=conf_thresh, verbose=False)
        annotated_frame = results[0].plot()

        # Count heads
        num_heads = len(results[0].boxes)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # ================== DRAW TEXT ==================
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        thickness = 2

        # Heads top-left
        cv2.putText(annotated_frame, f"Heads: {num_heads}", (10, 30),
                    font, 0.9, color, thickness)

        # FPS top-right
        text_fps = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(text_fps, font, 0.9, thickness)[0]
        cv2.putText(annotated_frame, text_fps,
                    (annotated_frame.shape[1] - text_size[0] - 10, 30),
                    font, 0.9, color, thickness)

        # Encode JPEG for browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Stream to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ================== ROUTES ==================
@app.route('/')
def index():
    return '''
    <html>
    <head><title>YOLO Stream</title></head>
    <body>
    <h2>YOLO Head Detection + FPS</h2>
    <img src="/video_feed" width="900">
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== MAIN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
