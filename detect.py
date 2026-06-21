# ============================================================
#  CCTV Criminal Detection — detect.py
# ============================================================

import cv2
import os
import json
import datetime
import threading
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- CONFIGURATION ---
ARCFACE_THRESHOLD = 0.50
PIXEL_THRESHOLD   = 70
STICKY_FRAMES     = 20   # Keeps a name label visible for ~0.7s after the last positive match
RECOG_EVERY       = 2    # Only run the (expensive) recognition model every 2nd frame
COOLDOWN_SEC      = 10   # Minimum gap between snapshots of the same person

DASHBOARD_HTML_PATH = "dashboard.html"

try:
    from deepface import DeepFace
    ARCFACE_AVAILABLE = True
    print("[ArcFace] DeepFace loaded successfully.")
except ImportError:
    ARCFACE_AVAILABLE = False
    print("[INFO] DeepFace not found. Using pixel-difference fallback.")

shared = {
    "detections": [],
    "stats": {
        "total_faces": 0,
        "total_criminals": 0,
        "total_unknown": 0,
        "session_start": "",
        "algorithm": "ArcFace" if ARCFACE_AVAILABLE else "Pixel Difference"
    },
    "latest_frame_jpg": b"",
    "status": "starting"
}
lock = threading.Lock()


# ============================================================
#  FACE TRACKING
# ============================================================
class FaceTracker:
    """
    detectMultiScale doesn't guarantee the same ordering of faces between
    frames, so indexing faces by their position in that frame's result list
    isn't a stable identity — two people can swap labels just by moving.
    This assigns each face a persistent ID by matching it to the closest
    tracked face from the previous frame (within max_distance pixels).
    A face that goes unmatched for too many frames in a row is dropped.
    """

    def __init__(self, max_distance=80, max_missed_frames=15):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 0
        self.tracked = {}  # id -> {centroid, missed, name, conf, sticky_left}

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def update(self, boxes):
        """Returns [(track_id, box), ...] for this frame's detections."""
        assignments = []
        used_ids = set()

        for box in boxes:
            cx, cy = self._centroid(box)
            best_id, best_dist = None, self.max_distance

            for tid, data in self.tracked.items():
                if tid in used_ids:
                    continue
                tx, ty = data["centroid"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracked[best_id] = {"name": None, "conf": 0, "sticky_left": 0}

            self.tracked[best_id]["centroid"] = (cx, cy)
            self.tracked[best_id]["missed"] = 0
            used_ids.add(best_id)
            assignments.append((best_id, box))

        for tid in list(self.tracked.keys()):
            if tid not in used_ids:
                self.tracked[tid]["missed"] += 1
                if self.tracked[tid]["missed"] > self.max_missed_frames:
                    del self.tracked[tid]

        return assignments

    def get_sticky(self, tid):
        data = self.tracked.get(tid)
        if data and data["sticky_left"] > 0:
            return data["name"], data["conf"]
        return None, 0

    def set_sticky(self, tid, name, conf, frames):
        if tid in self.tracked:
            self.tracked[tid]["name"] = name
            self.tracked[tid]["conf"] = conf
            self.tracked[tid]["sticky_left"] = frames

    def decay_sticky(self, tid):
        if tid in self.tracked and self.tracked[tid]["sticky_left"] > 0:
            self.tracked[tid]["sticky_left"] -= 1


# ============================================================
#  DATABASE & RECOGNITION
# ============================================================
def load_criminals(folder="database"):
    db = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[DB] '{folder}/' created. Add face photos inside.")
        return db

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(folder, filename)
        name = os.path.splitext(filename)[0].replace("_", " ").title()

        if ARCFACE_AVAILABLE:
            try:
                # Haar Cascade still does the fast per-frame face *detection* below,
                # but here DeepFace re-detects and aligns the face within the crop using
                # RetinaFace, which is far more accurate than Haar — better alignment
                # means a cleaner embedding and fewer false matches.
                result = DeepFace.represent(
                    img_path=path, model_name="ArcFace",
                    detector_backend="retinaface", enforce_detection=False
                )
                embedding = result[0]["embedding"]
                db.append({"name": name, "embedding": embedding, "path": path})
                print(f"[DB] Loaded (ArcFace): {name}")
            except Exception as e:
                print(f"[DB] Warning: {filename} — {e}")
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                db.append({"name": name, "photo": img, "path": path})
                print(f"[DB] Loaded (pixel): {name}")
    return db


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm_product) if norm_product != 0 else 0.0


def recognize_arcface(face_bgr, db):
    try:
        result = DeepFace.represent(
            img_path=face_bgr, model_name="ArcFace",
            detector_backend="retinaface", enforce_detection=False
        )
        live_emb = result[0]["embedding"]
        best_name, best_score = None, 0.0
        for c in db:
            sim = cosine_similarity(live_emb, c["embedding"])
            if sim > best_score:
                best_score = sim
                if sim >= ARCFACE_THRESHOLD:
                    best_name = c["name"]
        if best_name:
            # Rescale so the confidence shown to the user starts at 0% right at the
            # match threshold rather than at the raw (and less intuitive) cosine score.
            conf = int((best_score - ARCFACE_THRESHOLD) / (1.0 - ARCFACE_THRESHOLD) * 100)
            return best_name, max(0, min(99, conf))
    except Exception:
        pass
    return None, 0


def recognize_pixel(face_gray, db):
    best_name, best_score = None, 999
    face_r = cv2.resize(face_gray, (100, 100))
    for c in db:
        score = float(cv2.absdiff(face_r, cv2.resize(c["photo"], (100, 100))).mean())
        if score < best_score:
            best_score = score
            if score < PIXEL_THRESHOLD:
                best_name = c["name"]
    if best_name:
        return best_name, max(0, min(99, int((1 - best_score / PIXEL_THRESHOLD) * 100)))
    return None, 0


def save_snapshot(frame, name):
    os.makedirs("alerts", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"alerts/snap_{name.replace(' ', '')}_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path


def log_detection(face_id, name, is_criminal, conf, algo_label):
    """
    Logs one entry per "sighting" of a face rather than one per frame —
    a sighting continues until the name changes or 5+ seconds pass with
    no update, otherwise the same face would fill the log 30x a second.
    """
    now_dt = datetime.datetime.now()
    with lock:
        last_for_face = next(
            (d for d in reversed(shared["detections"]) if d.get("face_id") == face_id),
            None
        )
        should_log = (
            last_for_face is None
            or last_for_face["name"] != name
            or (now_dt - datetime.datetime.strptime(
                    last_for_face["date"] + " " + last_for_face["time"],
                    "%Y-%m-%d %H:%M:%S")
                ).seconds > 5
        )
        if should_log:
            shared["detections"].append({
                "face_id": face_id,
                "name": name,
                "is_criminal": is_criminal,
                "confidence": conf,
                "date": now_dt.strftime("%Y-%m-%d"),
                "time": now_dt.strftime("%H:%M:%S"),
                "algorithm": algo_label,
            })
            # The API only ever serves the most recent 40 entries, but without this
            # cap the underlying list would grow forever for as long as the server runs.
            if len(shared["detections"]) > 200:
                shared["detections"] = shared["detections"][-200:]
            shared["stats"]["total_faces"] += 1
            if is_criminal:
                shared["stats"]["total_criminals"] += 1
            else:
                shared["stats"]["total_unknown"] += 1


# ============================================================
#  WEB SERVER
# ============================================================
class Handler(BaseHTTPRequestHandler):

    def _path(self):
        # The browser appends a cache-busting timestamp (?12345) to /snapshot
        # and /data requests, so strip the query string before matching routes.
        return self.path.split("?")[0]

    def do_GET(self):
        p = self._path()

        if p in ("/", "/index.html"):
            try:
                with open(DASHBOARD_HTML_PATH, "rb") as f:
                    html = f.read()
            except FileNotFoundError:
                html = b"<h1>dashboard.html not found - place it next to detect.py</h1>"
            self._send(200, "text/html", html)

        elif p == "/snapshot":
            with lock:
                jpg = shared["latest_frame_jpg"]
            if jpg:
                self._send(200, "image/jpeg", jpg)
            else:
                self._send(503, "text/plain", b"No frame yet")

        elif p == "/data":
            with lock:
                body = json.dumps({
                    "detections": shared["detections"][-40:],
                    "stats": shared["stats"],
                    "status": shared["status"]
                }).encode()
            self._send(200, "application/json", body)

        else:
            self._send(404, "text/plain", b"Not Found")

    def _send(self, code, ct, body):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


# ============================================================
#  MAIN LOOP
# ============================================================
def run_detection():
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    criminals = load_criminals("database")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open the webcam. Is another app using it?")
        with lock:
            shared["status"] = "camera_error"
        return

    tracker = FaceTracker()
    algo_label = "ArcFace" if ARCFACE_AVAILABLE else "Pixel Difference"

    with lock:
        shared["status"] = "running"
        shared["stats"]["session_start"] = datetime.datetime.now().strftime("%H:%M:%S")

    last_alert = {}
    frame_n = 0

    while True:
        ok, frame = camera.read()
        if not ok:
            print("[WARN] Lost the camera feed, stopping.")
            break
        frame_n += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = face_detector.detectMultiScale(gray, 1.05, 4, minSize=(60, 60))
        tracked_faces = tracker.update(boxes)

        for face_id, (x, y, w, h) in tracked_faces:
            matched, conf = None, 0

            if frame_n % RECOG_EVERY == 0 and criminals:
                if ARCFACE_AVAILABLE:
                    matched, conf = recognize_arcface(frame[y:y + h, x:x + w], criminals)
                else:
                    matched, conf = recognize_pixel(gray[y:y + h, x:x + w], criminals)

                if matched:
                    tracker.set_sticky(face_id, matched, conf, STICKY_FRAMES)

            sticky_name, sticky_conf = tracker.get_sticky(face_id)
            if sticky_name:
                matched, conf = sticky_name, sticky_conf
            tracker.decay_sticky(face_id)

            is_criminal = matched is not None
            name = matched if matched else "Unknown"
            color = (0, 0, 220) if is_criminal else (0, 200, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3 if is_criminal else 2)
            cv2.putText(frame, f"{name} {conf}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_criminal:
                now = datetime.datetime.now()
                if name not in last_alert or (now - last_alert[name]).seconds > COOLDOWN_SEC:
                    last_alert[name] = now
                    save_snapshot(frame, name)

            log_detection(face_id, name, is_criminal, conf, algo_label)

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            shared["latest_frame_jpg"] = jpg.tobytes()

        cv2.imshow("CCTV (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    with lock:
        shared["status"] = "stopped"


if __name__ == "__main__":
    print("[SERVER] Dashboard → http://localhost:5000")
    threading.Thread(
        target=lambda: HTTPServer(("", 5000), Handler).serve_forever(),
        daemon=True
    ).start()
    run_detection()