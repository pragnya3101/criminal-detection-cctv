# 🎥 Real-Time Face Recognition & Tracking Dashboard

A face recognition system using a webcam, OpenCV, and DeepFace (ArcFace). Detected faces are matched against a local face database and shown on a web dashboard that refreshes automatically (polling every ~1 second).

---

## ✨ Features

- 🔍 **ArcFace face recognition** — ResNet-50, 512-D embeddings, cosine similarity matching, with RetinaFace alignment for cleaner embeddings
- 🆔 **Stable per-face tracking** — faces keep the same identity across frames via centroid tracking, instead of relying on detection order
- 📷 **Webcam feed** — dashboard polls the latest frame roughly once per second
- ⚡ **Near-instant alerts** — red flash overlay appears within ~1 second of a match (on the next dashboard poll)
- 📋 **Detection log** — timestamped cards with name, confidence, and algorithm used
- 💾 **Snapshot saving** — auto-saves a photo to `alerts/` on a positive match
- 🔄 **Pixel-difference fallback** — works even without DeepFace installed
- 🌐 **Built-in web server** — no Flask or external frameworks needed
- 📊 **Offline evaluation script** — measure precision/recall/F1 against a labeled test set

---

## 🛠️ Tech Stack

| Layer             | Technology                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| Face Detection    | OpenCV Haar Cascade (per-frame) + RetinaFace re-alignment (recognition step) |
| Face Recognition  | DeepFace · ArcFace (ResNet-50)                                               |
| Tracking (live)   | Custom centroid-based tracker (`FaceTracker` in `detect.py`)                 |
| Backend           | Python · threading · http.server                                            |
| Frontend          | HTML · CSS · Vanilla JavaScript                                             |
| Similarity Metric | Cosine Similarity                                                            |
| Evaluation        | scikit-learn (classification report, confusion matrix)                     |

---

## 📁 Project Structure

```
face-recognition-tracking-dashboard/
│
├── detect.py            # Detection, recognition, tracking, and web server (the live pipeline)
├── Sorttracker.py        # Standalone SORT-style tracker (Kalman filter + Hungarian matching).
│                         #   See "Project Status" below — not yet wired into detect.py.
├── Evaluate.py            # Offline evaluation script — runs the recognition pipeline
│                         #   against a labeled test set and outputs a classification
│                         #   report + confusion matrix
├── dashboard.html        # Live web dashboard UI
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore
├── screenshots/
├── database/              # Add your own face photos here (.jpg/.png) — created automatically on first run
├── alerts/                # Snapshots of matched faces — created automatically on first run
├── test_set/              # (you create this) Labeled images for Evaluate.py — see below
└── eval_results/           # Output of Evaluate.py — created automatically when you run it
```
> `database/`, `alerts/`, and `eval_results/` aren't tracked in this repo — they're created automatically the first time you run the relevant script.

---

## 🚀 Getting Started

### 1. Clone the repository

```
git clone https://github.com/pragnya3101/criminal-detection-cctv.git
cd criminal-detection-cctv
```

### 2. Install dependencies

```
pip install -r requirements.txt
```
> **Note:** DeepFace will auto-download the ArcFace model weights on first run (~200 MB).
> If DeepFace isn't installed, the system automatically falls back to pixel-difference comparison.
> `requirements.txt` covers the live detection pipeline (`detect.py`). The same file also includes
> `scipy`, `scikit-learn`, and `matplotlib`, which are only needed if you plan to run `Evaluate.py`
> or experiment with `Sorttracker.py`.

### 3. Add face photos to the database

```
database/
├── john_doe.jpg
├── jane_smith.png
└── ...
```

- One clear, front-facing photo per person
- The filename becomes the displayed name (underscores → spaces, title-cased)

### 4. Run the system

```
python detect.py
```

### 5. Open the dashboard

```
http://localhost:5000
```

---

## 📊 Evaluating Recognition Accuracy

`Evaluate.py` runs the same detection + recognition pipeline used by `detect.py` against a
folder of labeled test images, instead of a live webcam feed. Useful for getting real
precision/recall numbers instead of eyeballing the dashboard.

### 1. Set up a labeled test set

```
test_set/
├── labels.csv          # columns: filename, true_identity
├── john_at_angle.jpg
├── jane_dim_light.jpg
└── ...
```

`labels.csv` example:
```
filename,true_identity
john_at_angle.jpg,John Doe
jane_dim_light.jpg,Jane Smith
random_stranger.jpg,unknown
```

### 2. Run it

```
python Evaluate.py --test-dir test_set --labels test_set/labels.csv --out-dir eval_results --db-dir database
```

This loads the same `database/` folder `detect.py` uses, runs detection + recognition on each
test image, and writes:
- `eval_results/report.txt` — precision/recall/F1 per identity
- `eval_results/confusion_matrix.png`
- `eval_results/errors.csv` — every misclassified filename with true vs. predicted identity

---

## Demo

[![Dashboard](https://github.com/pragnya3101/criminal-detection-cctv/raw/main/screenshots/dashboard.png)](/pragnya3101/criminal-detection-cctv/blob/main/screenshots/dashboard.png)

---

## ⚙️ Configuration

Edit these constants at the top of `detect.py`:

| Variable            | Default | Description                                                             |
| -------------------- | ------- | ------------------------------------------------------------------------ |
| `ARCFACE_THRESHOLD` | `0.50`  | Cosine similarity cutoff for a match                                    |
| `PIXEL_THRESHOLD`   | `70`    | Pixel-difference cutoff (fallback mode)                                |
| `STICKY_FRAMES`     | `20`    | Frames a name label stays visible after the last positive match (~0.7s) |
| `RECOG_EVERY`       | `2`     | Run recognition every Nth frame                                        |
| `COOLDOWN_SEC`      | `10`    | Minimum seconds between snapshots of the same person                   |

---

## 🧠 How It Works

1. **Face detection** — Haar Cascade scans each webcam frame for faces (`detect.detect_faces`)
2. **Tracking** — each detected face is matched to the closest tracked face from the previous frame, giving it a stable ID across frames instead of relying on detection order (`FaceTracker`)
3. **Feature extraction** — DeepFace re-aligns each tracked face with RetinaFace, then extracts a 512-dimensional ArcFace embedding from the aligned crop
4. **Matching** — cosine similarity is computed against every embedding in the local database
5. **Alert** — if similarity exceeds the threshold, the face is flagged, a snapshot is saved, and the dashboard shows a red alert
6. **Dashboard** — the browser polls `/snapshot` (camera feed) and `/data` (detections + stats) every second

---

## 📌 Project Status / Known Gaps

Being upfront about where this stands, beyond the recognition-accuracy limitations below:

- **`Sorttracker.py` is a complete, independently-functional SORT-style tracker** (Kalman filter
  with a constant-velocity model + Hungarian-algorithm IOU matching) but it is **not currently
  used by `detect.py`**. The live pipeline uses the simpler centroid-distance `FaceTracker`
  instead. Wiring `Sort` in as the primary tracker (it would need an adapter to convert Haar's
  `(x, y, w, h)` boxes to the `[x1, y1, x2, y2]` format `Sort` expects) is on the roadmap —
  it should handle fast motion and brief occlusion better than centroid matching.
- **`Evaluate.py` works end-to-end** but has no bundled test set — you need to assemble your
  own `test_set/` + `labels.csv` to get real numbers (see above).
- **No automated tests yet** for the core matching/tracking logic.
- **ArcFace embeddings are recomputed from images every time `detect.py` or `Evaluate.py`
  starts** — there's no on-disk caching, so startup time scales with database size.
- **Camera source is hardcoded to the default webcam (`cv2.VideoCapture(0)`)** — no config
  option yet for an IP camera / RTSP stream, despite the dashboard styling itself as CCTV-style.

---

## ⚠️ Known Limitations

- Haar Cascade struggles with non-frontal faces, low light, and partial occlusion (masks, hats, side profiles)
- No liveness detection — a printed photo or screen image can be matched against the database
- Accuracy depends heavily on the quality and quantity of reference photos per person
- Centroid-based tracking can lose or merge identities if two faces cross paths or move very quickly
- Built for local, single-camera use — not tested for multi-camera or networked deployment
- This is a learning project, not a production-grade or legally compliant surveillance system. Running it against real people without their knowledge or consent has real privacy and legal implications.

---

## 👩‍💻 Author

**Pragnya Bodakuntla**
B.Tech CSE(DS) | Hyderabad, India
[GitHub](https://github.com/pragnya3101) · [LinkedIn](https://www.linkedin.com/in/pragnya-bodakuntla)
