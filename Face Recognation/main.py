# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import urllib.request
import time
import shutil

st.set_page_config(page_title="Offline Face Recognition (Haar + DNN + LBPH/ORB)", layout="wide")
st.title("Offline Face Recognition — Haar & DNN detectors (No DeepFace)")

# ----------------------
# Directories & files
# ----------------------
REF_DIR = "reference_images"
CASCADE_DIR = "cascades"
DNN_DIR = "face_detector"
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(CASCADE_DIR, exist_ok=True)
os.makedirs(DNN_DIR, exist_ok=True)

# DNN files (auto-download if missing)
proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
dnn_proto = os.path.join(DNN_DIR, "deploy.prototxt")
dnn_model = os.path.join(DNN_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

def try_download(url, path):
    if os.path.exists(path):
        return True
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        st.warning(f"Could not download {os.path.basename(path)}: {e}")
        return False

_ = try_download(proto_url, dnn_proto)
_ = try_download(model_url, dnn_model)

# Haar cascade (try using local cv2 data, otherwise download)
haar_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
builtin_haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if os.path.exists(builtin_haar):
    if not os.path.exists(haar_path):
        try:
            shutil.copy(builtin_haar, haar_path)
        except:
            pass
else:
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    _ = try_download(haar_url, haar_path)

# ----------------------
# Utilities
# ----------------------
def load_image(file):
    try:
        img = Image.open(file)
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        st.warning(f"Cannot process {getattr(file,'name',file)}: {e}")
        return None

# Load detectors
haar_cascade = cv2.CascadeClassifier(haar_path) if os.path.exists(haar_path) else cv2.CascadeClassifier()
dnn_net = None
if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
    except Exception as e:
        st.warning(f"Failed to load DNN detector: {e}")

if haar_cascade.empty():
    st.error("Haar cascade failed to load. Make sure haarcascade xml exists in cascades folder.")
if dnn_net is None:
    st.info("DNN model not available — DNN option will be disabled.")

def detect_faces(img_rgb, method="haar", conf_threshold=0.5):
    """Return list of boxes (x,y,w,h) in RGB image coordinates"""
    faces = []
    h, w = img_rgb.shape[:2]
    if method == "haar":
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        faces = [tuple(r) for r in rects]
    elif method == "dnn":
        if dnn_net is None:
            return []
        blob = cv2.dnn.blobFromImage(cv2.resize(img_rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

def draw_annotations(img_rgb, boxes, labels, confidences=None):
    img = img_rgb.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x, y, w, h = box
        if label and label != "Unknown":
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{label}"
            if confidences and confidences[i] is not None:
                text += f" ({confidences[i]:.1f})"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return img

# ----------------------
# Recognition backend: try LBPH, fallback to ORB
# ----------------------
use_lbph = False
recognizer = None
orb_reference = []  # list of tuples (name, kp, des)
bf = None

# Try to create LBPH recognizer (requires opencv-contrib-python)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    use_lbph = True
except Exception:
    use_lbph = False
    recognizer = None

# ----------------------
# Admin uploads reference images
# ----------------------
st.sidebar.header("Admin: Upload reference images (one person per file recommended)")
uploaded_files = st.sidebar.file_uploader(
    "Upload reference images (multiple)", 
    type=['jpg','jpeg','png','bmp','tiff','gif'], 
    accept_multiple_files=True
)

reference_paths = []
reference_names = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        img_np = load_image(uploaded_file)
        if img_np is None:
            continue
        save_name = uploaded_file.name
        save_path = os.path.join(REF_DIR, save_name)
        Image.fromarray(img_np).save(save_path)
        reference_paths.append(save_path)
        reference_names.append(os.path.splitext(save_name)[0])

# Also include existing files in REF_DIR
for f in os.listdir(REF_DIR):
    fp = os.path.join(REF_DIR, f)
    if fp not in reference_paths and os.path.isfile(fp):
        reference_paths.append(fp)
        reference_names.append(os.path.splitext(f)[0])

st.sidebar.write(f"{len(reference_paths)} reference image(s) available.")

# ----------------------
# Build recognizer from references
# ----------------------
detector_choice_for_training = st.sidebar.selectbox("Detector used to preprocess reference faces (for training):", ["haar", "dnn"] if dnn_net is not None else ["haar"])

# training parameters
FACE_SIZE = (200, 200)

def prepare_face_crop(img_rgb, box):
    x,y,w,h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_rgb.shape[1], x + w), min(img_rgb.shape[0], y + h)
    face = img_rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    try:
        resized = cv2.resize(gray, FACE_SIZE)
    except Exception:
        return None
    return resized

def build_recognizer():
    global recognizer, use_lbph, orb_reference, bf
    images = []
    labels = []
    orb_reference = []
    label_to_name = {}
    current_label = 0

    if len(reference_paths) == 0:
        return None, None  # nothing to train on

    for path, name in zip(reference_paths, reference_names):
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img, dtype=np.uint8)
        except Exception as e:
            st.warning(f"Skipping {path}: {e}")
            continue

        # detect faces in reference image using chosen detector
        boxes = detect_faces(img_np, method=detector_choice_for_training)
        if len(boxes) == 0:
            # try fallback: whole image as face
            h,w = img_np.shape[:2]
            boxes = [(0,0,w,h)]

        # We'll take the largest detected face (common case)
        boxes_sorted = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        face_crop = prepare_face_crop(img_np, boxes_sorted[0])
        if face_crop is None:
            st.warning(f"Could not extract face from {path}")
            continue

        if use_lbph:
            images.append(face_crop)
            labels.append(current_label)
            label_to_name[current_label] = name
            current_label += 1
        else:
            # ORB fallback: compute keypoints/descriptors
            orb = cv2.ORB_create(nfeatures=500)
            kp, des = orb.detectAndCompute(face_crop, None)
            if des is None:
                st.warning(f"ORB could not compute descriptors for {name}, skipping.")
                continue
            orb_reference.append((name, kp, des))
    # Train LBPH if available
    if use_lbph and len(images) > 0:
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels))
            return recognizer, label_to_name
        except Exception as e:
            st.warning(f"LBPH training failed, switching to ORB fallback: {e}")
            # switch to ORB
            use_lbph = False
            recognizer = None

    if not use_lbph:
        # prepare BF matcher for ORB
        if len(orb_reference) == 0:
            return None, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        return ("ORB", orb_reference)

    return None, None

train_button = st.sidebar.button("Build/Train Recognizer from references")
trained_info = None
if train_button:
    info = build_recognizer()
    if info is None or (use_lbph and info[0] is None):
        st.sidebar.error("No valid faces found in reference images. Upload clearer face images and retry.")
    else:
        st.sidebar.success("Recognizer built successfully.")
        trained_info = info

# Automatically build recognizer if there are references and recognizer not built yet
if len(reference_paths) > 0 and trained_info is None:
    trained_info = build_recognizer()

# ----------------------
# Helper: match face crop
# ----------------------
def match_face(face_rgb):
    """
    face_rgb: RGB full-color face crop (not gray)
    returns (matched_bool, name_or_None, confidence_or_None)
    """
    if face_rgb is None or face_rgb.size == 0:
        return False, None, None
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    face_resized = cv2.resize(gray, FACE_SIZE)
    # LBPH
    if use_lbph and recognizer is not None:
        try:
            label, confidence = recognizer.predict(face_resized)
            # LBPH confidence: lower == better. threshold around 60-80 typical, adjust as needed.
            if confidence < 80:
                return True, trained_info[1][label] if isinstance(trained_info, tuple) else f"ID_{label}", confidence
            else:
                return False, None, confidence
        except Exception as e:
            return False, None, None
    else:
        # ORB fallback
        if not orb_reference or bf is None:
            return False, None, None
        orb = cv2.ORB_create(nfeatures=500)
        kp2, des2 = orb.detectAndCompute(face_resized, None)
        if des2 is None:
            return False, None, None
        best_name = None
        best_score = 0
        for (name, kp1, des1) in orb_reference:
            if des1 is None:
                continue
            try:
                matches = bf.match(des1, des2)
            except Exception:
                continue
            if not matches:
                continue
            # sort matches by distance (lower = better)
            matches = sorted(matches, key=lambda x: x.distance)
            # compute a simple score: number of good matches under a distance threshold
            good = [m for m in matches if m.distance < 60]
            score = len(good)
            if score > best_score:
                best_score = score
                best_name = name
        # threshold of good matches -> consider a match, tweak as needed
        if best_score >= 10:
            # convert score to pseudo-confidence: higher better
            return True, best_name, best_score
        return False, None, None

# ----------------------
# UI Controls for recognition
# ----------------------
st.header("Face Recognition")
detector_options = ["haar"]
if dnn_net is not None:
    detector_options.append("dnn")
detector_choice = st.selectbox("Choose Face Detector (used at detection time)", detector_options)
option = st.radio("Choose input method:", ["Upload Test Image", "Camera"])

# ----------------------
# Upload Test Image flow
# ----------------------
if option == "Upload Test Image":
    test_file = st.file_uploader("Upload a test image", type=['jpg','jpeg','png','bmp','tiff','gif'])
    if test_file:
        test_img_np = load_image(test_file)
        if test_img_np is not None:
            if len(reference_paths) == 0 or (use_lbph and recognizer is None and trained_info is None):
                st.warning("No trained references available. Upload reference images and build recognizer.")
            boxes = []
            labels = []
            confidences = []
            faces = detect_faces(test_img_np, method=detector_choice)
            if len(faces) == 0:
                st.info("No faces detected in test image.")
                st.image(test_img_np, caption="No faces detected")
            else:
                for (x,y,w,h) in faces:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(test_img_np.shape[1], x + w), min(test_img_np.shape[0], y + h)
                    face_crop = test_img_np[y1:y2, x1:x2]
                    matched, name, conf = match_face(face_crop)
                    boxes.append((x1,y1,x2-x1,y2-y1))
                    if matched:
                        labels.append(name)
                        confidences.append(conf)
                    else:
                        labels.append("Unknown")
                        confidences.append(None)
                annotated = draw_annotations(test_img_np, boxes, labels, confidences)
                st.image(annotated, caption="Detections (green=match, red=unknown)", use_column_width=True)

# ----------------------
# Camera flow (single-frame capture)
# ----------------------
elif option == "Camera":
    start_camera = st.checkbox("Start Camera (capture one frame)")
    if "cam" not in st.session_state:
        st.session_state.cam = None

    if start_camera and st.session_state.cam is None:
        st.session_state.cam = cv2.VideoCapture(0)
        time.sleep(0.5)

    if st.session_state.cam is not None and start_camera:
        cam = st.session_state.cam
        if not cam.isOpened():
            st.error("Could not open webcam. Check permissions.")
            cam.release()
            st.session_state.cam = None
        else:
            ret, frame = cam.read()
            if not ret:
                st.warning("Failed to read frame from camera.")
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if len(reference_paths) == 0:
                    st.warning("No reference images loaded. Upload references and build recognizer first.")
                    st.image(rgb_frame)
                else:
                    faces = detect_faces(rgb_frame, method=detector_choice)
                    if len(faces) == 0:
                        st.info("No faces detected in camera frame.")
                        st.image(rgb_frame)
                    else:
                        boxes = []
                        labels = []
                        confidences = []
                        for (x,y,w,h) in faces:
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(rgb_frame.shape[1], x + w), min(rgb_frame.shape[0], y + h)
                            face_crop = rgb_frame[y1:y2, x1:x2]
                            matched, name, conf = match_face(face_crop)
                            boxes.append((x1,y1,x2-x1,y2-y1))
                            if matched:
                                labels.append(name)
                                confidences.append(conf)
                            else:
                                labels.append("Unknown")
                                confidences.append(None)
                        annotated = draw_annotations(rgb_frame, boxes, labels, confidences)
                        st.image(annotated)
    stop_button = st.button("Stop Camera")
    if stop_button and st.session_state.get("cam") is not None:
        try:
            st.session_state.cam.release()
        except:
            pass
        st.session_state.cam = None
        st.success("Camera stopped.")

# ----------------------
# Footer / hints
# ----------------------
st.markdown("---")
st.markdown(
    """
    **Notes & tips**
    - This app uses **Haar cascade** and **OpenCV DNN (res10 SSD)** strictly for **detection**.  
    - Recognition uses **LBPH** (recommended — faster & robust) if `cv2.face` is available (OpenCV contrib).  
      If LBPH is not available, it falls back to an **ORB + BFMatcher** approach.
    - For best results, upload clear, frontal reference face images (one person per file is easiest).
    - You may need to install `opencv-contrib-python` to get `cv2.face` (LBPH) support:
      `pip install opencv-contrib-python`
    - Adjust thresholds in the code (LBPH confidence threshold or ORB matching threshold) if you need more/less strict matching.
    """
)
