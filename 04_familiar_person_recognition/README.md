# Feature 4: Familiar Person & Face Recognition System

> **Sub-system of the AI Assistive Vision System for Visually Impaired Users**  
> Real-time deep learning facial detection, metric feature embedding extraction, one-shot enrollment, and spatial proximity audio feedback.

---

## 1. System Overview

For visually impaired individuals, identifying who is approaching or present in a room is a major accessibility challenge. Traditional computer vision relies on complex re-training per person. This system utilizes **Deep Metric Learning** to extract **128-dimensional mathematical face embeddings**, enabling instant **one-shot enrollment** directly from photo folders.

```
 ┌────────────────────────────────────────────────────────┐
 │            1. Offline Photo Enrollment                 │
 │  known_faces/ [Merinson/, Mithul/, Sandra/, Sourav/]  │
 │                      ↓                                 │
 │   Deep Face Embedding Extraction (128-D Vector)        │
 │                      ↓                                 │
 │     Vector Database Cache (`face_database.pkl`)        │
 └──────────────────────┬─────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │           2. Real-Time Camera Stream Inference         │
 │                      ↓                                 │
 │   Face Detection & Landmark Alignment (YuNet / SFace)  │
 │                      ↓                                 │
 │       Extract Live 128-D Face Vector Embedding         │
 │                      ↓                                 │
 │   Cosine Metric Matcher: Score = (A · B) / (||A||*||B||) │
 │                      ↓                                 │
 │        Match Threshold Gate (Confidence >= 75%)        │
 │       ├── Match Found  ──> Name (e.g., "Merinson")     │
 │       └── No Match     ──> "Unknown Person"            │
 │                      ↓                                 │
 │   Spatial Corridor Analysis (Left / Center / Right)    │
 │   Proximity Estimation (Face Bounding Box Height)      │
 │                      ↓                                 │
 │ 3. Positional Audio Cue:                               │
 │    "Merinson is 1.5 meters ahead of you."              │
 │    "Unknown person detected on your left."             │
 └────────────────────────────────────────────────────────┘
```

---

## 2. Technical Architecture & Algorithms

### A. Face Detection & 5-Point Landmark Alignment (YuNet)
- **Model**: Lightweight CNN Face Detector (`YuNet`, ~300KB) operating natively via OpenCV DNN.
- **Output**: Bounding box $[x, y, w, h]$, detection confidence, and 5 key facial landmarks:
  1. Right Eye Center
  2. Left Eye Center
  3. Nose Tip
  4. Right Mouth Corner
  5. Left Mouth Corner
- **Alignment**: Using affine transformation, the face is rotated and scaled to a canonical normalized $112 \times 112$ pose (`alignCrop`), eliminating distortions from head tilts.

### B. Deep Metric Learning & 128-D Embeddings (SFace)
- **Model**: `SFace` (SphereFace / Cosine-Loss Deep Neural Network, ~3.8MB).
- **Embeddings**: Transforms raw pixel data into a normalized unit vector $\vec{v} \in \mathbb{R}^{128}$ on a hypersphere.
- **Centroid Aggregation**: When multiple photos of the same person are provided (e.g. 10 photos in `Merinson/`), their feature vectors are averaged and re-normalized:
  $$\vec{v}_{\text{centroid}} = \frac{\sum_{i=1}^{N} \vec{v}_i}{\left\| \sum_{i=1}^{N} \vec{v}_i \right\|}$$

### C. Cosine Distance Metric Matching
Given a live face embedding vector $\vec{A}$ and enrolled vector $\vec{B}$:
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
- **Threshold**: Values $\ge 0.42$ indicate an authentic identity match.

### D. Pin-hole Distance & Spatial Estimation
Using camera optical geometry:
$$\text{Distance (meters)} = \frac{\text{Real Face Height (0.20m)} \times \text{Focal Length (550 px)}}{\text{Detected Bounding Box Height (px)}}$$

---

## 3. Directory Structure

```text
d:\Ashik\project\
├── known_faces/                      <-- Enrolled Photo Library
│   ├── Merinson/                     (10 images - JPG + HEIC)
│   ├── Mithul/                       (10 images - JPG)
│   ├── Sandra/                       (11 images - JPG + HEIC)
│   └── Sourav/                       (10 images - JPG + HEIC)
├── 04_familiar_person_recognition/
│   ├── models/                       (YuNet & SFace ONNX models)
│   ├── config.py                     (System hyperparameters & thresholds)
│   ├── face_engine.py                (YuNet detector + SFace embedding extractor)
│   ├── face_database.py              (Multi-photo encoder & pickle cache manager)
│   ├── face_voice.py                 (Spatial audio announcer with anti-chatter cooldown)
│   ├── enroll_faces.py               (1-click standalone enrollment utility)
│   ├── main.py                       (Standalone feature runner with live AR HUD)
│   └── README.md                     (Full documentation + Viva Voce Q&As)
```

---

## 4. How to Run

### Standalone Feature Run:
```powershell
python 04_familiar_person_recognition/main.py
```

### Re-enroll / Add New People:
1. Create a new folder inside `known_faces/` (e.g. `known_faces/Rahul/`) and paste 1 or more photos.
2. Run the enrollment tool:
```powershell
python 04_familiar_person_recognition/enroll_faces.py
```

### Hotkey Controls:
- `[R]` - Reload and rescan face photos in real time.
- `[V]` - Toggle Voice Guidance (Mute / Unmute).
- `[Q]` / `[ESC]` - Exit.

---

## 5. Viva Voce Examination Questions & Answers (10 Q&As)

### Q1: What is Deep Metric Learning in Face Recognition?
**Answer:** Deep Metric Learning is a technique where a neural network is trained using contrastive or triplet loss to map high-dimensional facial images into a low-dimensional Euclidean space (128-D vector) where distances directly correspond to facial similarity (faces of the same person are clustered close together, while different people are pushed far apart).

### Q2: Why is one-shot embedding superior to traditional machine learning classification (like SVM/Softmax) for face recognition?
**Answer:** Traditional classification requires re-training and re-optimizing the entire neural network every time a new person is added. With one-shot metric embeddings, adding a new identity requires only a single forward pass to compute and store a 128-D vector, taking less than 1 second.

### Q3: Why is 5-point facial landmark alignment essential before feature extraction?
**Answer:** Facial alignment uses landmark positions (both eyes, nose tip, mouth corners) to apply an affine transform. This standardizes face orientation, scale, and tilt to a canonical $112 \times 112$ frame, making feature extraction robust against tilted or rotated head poses.

### Q4: How does Cosine Similarity differ from Euclidean Distance in face verification?
**Answer:** Because deep face embedding vectors are normalized to have unit magnitude ($\|\vec{v}\| = 1$), Cosine Similarity measures the angle between vectors on a hypersphere ($\cos \theta = \vec{A} \cdot \vec{B}$), which is mathematically equivalent to Euclidean distance ($D = \sqrt{2(1 - \cos \theta)}$) but computationally faster.

### Q5: How does multi-image centroid aggregation improve recognition accuracy?
**Answer:** By taking multiple photos of a person under different lighting conditions and angles, computing the centroid $\vec{v}_{\text{centroid}} = \text{norm}(\sum \vec{v}_i)$ filters out noise, shadows, and expression variations, creating a robust master identity template.

### Q6: How is real-time distance estimated without an expensive hardware LiDAR or stereo camera?
**Answer:** We utilize the geometric pin-hole camera model: $\text{Distance} = \frac{\text{Real Height} \times f}{\text{Pixel Height}}$. Since the adult human face height is consistently $\approx 20\text{ cm}$, the ratio of detected bounding box height to camera focal length yields an accurate real-time distance estimate.

### Q7: How does the system handle Apple iPhone `.HEIC` image formats alongside standard `.JPG`?
**Answer:** Through the integrated `pillow-heif` library, the enrollment pipeline natively decodes Apple High-Efficiency Image Format (HEIC) bitstreams directly into RGB NumPy arrays without requiring external file conversion.

### Q8: What prevents audio overload when a visually impaired user is having a long conversation with a recognized person?
**Answer:** The `FaceVoiceAssistant` enforces a temporal cooldown timer ($\tau = 8.0\text{ seconds}$) per recognized identity. Once announced, subsequent repetitive announcements for that specific person are suppressed until the cooldown window expires.

### Q9: What happens when an unregistered person enters the field of view?
**Answer:** If the highest cosine similarity score against all enrolled centroids is below the threshold ($\theta < 0.42$), the system classifies the face as `"Unknown"` and emits a spatial alert (e.g., *"Unknown person on your left"*).

### Q10: What are the primary performance metrics of this pipeline?
**Answer:**
- **Inference Speed**: ~30+ FPS (YuNet detector takes ~10ms, SFace feature extraction takes ~12ms on CPU).
- **Embedding Footprint**: 128 float32 values = 512 bytes per face.
- **Startup Time**: Instantaneous (< 50ms) using pickled vector caches.
