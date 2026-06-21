

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Reuse the actual recognition pipeline from the project instead of
# reimplementing matching logic here — we want to evaluate the real system.
import detect  # noqa: E402  (detect.py lives at repo root)


UNKNOWN_LABEL = "unknown"


def load_labels(labels_csv):
    """labels.csv: filename,true_identity"""
    rows = []
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filename"], row["true_identity"].strip()))
    return rows


def predict_identity(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    faces = detect.detect_faces(img)  # Haar cascade detection
    if not faces:
        return UNKNOWN_LABEL

    # Evaluate on the largest detected face (most images in a test set
    # of this kind are single-subject).
    face_box = max(faces, key=lambda b: b[2] * b[3])
    name, score = detect.recognize_face(img, face_box)  # ArcFace + cosine sim

    if name is None or score < detect.ARCFACE_THRESHOLD:
        return UNKNOWN_LABEL
    return name


def run_evaluation(test_dir, labels_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_labels(labels_csv)

    y_true, y_pred, filenames = [], [], []
    errors = []

    for filename, true_label in rows:
        path = os.path.join(test_dir, filename)
        try:
            pred = predict_identity(path)
        except Exception as e:
            print(f"[skip] {filename}: {e}", file=sys.stderr)
            continue

        y_true.append(true_label)
        y_pred.append(pred)
        filenames.append(filename)

        if pred != true_label:
            errors.append((filename, true_label, pred))

    labels_sorted = sorted(set(y_true) | set(y_pred))

    print("\n=== Classification report ===")
    report = classification_report(y_true, y_pred, labels=labels_sorted, zero_division=0)
    print(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0
    )
    print(f"Macro precision: {precision:.3f} | Macro recall: {recall:.3f} | Macro F1: {f1:.3f}")
    print(f"n = {len(y_true)} images, {len(labels_sorted)} classes (incl. '{UNKNOWN_LABEL}')")

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("\n=== Confusion matrix ===")
    print("Labels:", labels_sorted)
    print(cm)

    _save_confusion_matrix_png(cm, labels_sorted, os.path.join(out_dir, "confusion_matrix.png"))

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(out_dir, "errors.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_identity", "predicted_identity"])
        writer.writerows(errors)

    print(f"\nSaved: {out_dir}/confusion_matrix.png, {out_dir}/report.txt, {out_dir}/errors.csv")
    print(f"{len(errors)}/{len(y_true)} misclassified — see errors.csv for the exact images.")


def _save_confusion_matrix_png(cm, labels, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.8), max(4, len(labels) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Face Recognition")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate face recognition pipeline against a labeled test set.")
    parser.add_argument("--test-dir", default="test_set", help="Directory containing test images")
    parser.add_argument("--labels", default="test_set/labels.csv", help="CSV with filename,true_identity")
    parser.add_argument("--out-dir", default="eval_results", help="Where to write report/confusion matrix")
    args = parser.parse_args()

    run_evaluation(args.test_dir, args.labels, args.out_dir)