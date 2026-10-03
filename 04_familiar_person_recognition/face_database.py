"""
Face Database & Multi-Image Enrollment Manager.
Extracts, aggregates, and caches deep 128-D feature embeddings for known persons.
"""

import os
import glob
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from . import config
    from .face_engine import FaceEngine, load_image_robust
except ImportError:
    import config
    from face_engine import FaceEngine, load_image_robust

class FaceDatabase:
    def __init__(self, engine: Optional[FaceEngine] = None):
        self.engine = engine if engine is not None else FaceEngine()
        # Storage: { "PersonName": [list of 128-D vectors] }
        self.database: Dict[str, List[np.ndarray]] = {}
        # Precomputed centroid vectors: { "PersonName": normalized_128D_centroid }
        self.centroids: Dict[str, np.ndarray] = {}
        
        self.load_or_enroll()

    def load_or_enroll(self, force_rebuild: bool = False):
        """Loads cached embeddings from disk or scans known_faces directory to build it."""
        if not force_rebuild and os.path.exists(config.DATABASE_CACHE_PATH):
            try:
                with open(config.DATABASE_CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.database = data.get("database", {})
                    self.centroids = data.get("centroids", {})
                    print(f"[Face Database] Loaded {len(self.centroids)} enrolled identities from cache.")
                    for name, vecs in self.database.items():
                        print(f"  * {name}: {len(vecs)} enrolled photo embeddings")
                    return
            except Exception as e:
                print(f"[Face Database] Cache read failed ({e}), rebuilding database...")

        self.rebuild_database()

    def rebuild_database(self):
        """Scans known_faces folder, reads all images, extracts deep vectors, and saves cache."""
        print("=" * 65)
        print("   [SCAN] ENROLLING IDENTITIES FROM KNOWN_FACES DATASET")
        print("=" * 65)
        print(f"Scanning directory: {config.KNOWN_FACES_DIR}\n")

        self.database = {}
        self.centroids = {}

        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
            print(f"[Face Database] Created empty folder: {config.KNOWN_FACES_DIR}")
            return

        # 1. Scan subdirectories (e.g. known_faces/Merinson/, known_faces/Sandra/)
        subdirs = [d for d in os.listdir(config.KNOWN_FACES_DIR) if os.path.isdir(os.path.join(config.KNOWN_FACES_DIR, d))]
        
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".heic")

        for person_name in subdirs:
            person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
            image_files = [
                os.path.join(person_dir, f) for f in os.listdir(person_dir)
                if f.lower().endswith(valid_exts)
            ]
            if not image_files:
                continue

            print(f"Processing Person: [{person_name}] ({len(image_files)} photos)...")
            person_vectors = []
            for img_path in image_files:
                img = load_image_robust(img_path)
                if img is None:
                    continue
                
                vec = self.engine.extract_single_face_vector(img)
                if vec is not None:
                    person_vectors.append(vec)

            if person_vectors:
                self.database[person_name] = person_vectors
                # Compute centroid vector
                avg_vec = np.mean(person_vectors, axis=0)
                norm_avg = avg_vec / max(1e-6, np.linalg.norm(avg_vec))
                self.centroids[person_name] = norm_avg
                print(f"  + Successfully encoded {len(person_vectors)}/{len(image_files)} faces for '{person_name}'.")
            else:
                print(f"  [!] Warning: No faces could be detected in photos for '{person_name}'.")

        # 2. Scan loose files in known_faces (e.g. known_faces/Ashik.jpg)
        loose_files = [
            f for f in os.listdir(config.KNOWN_FACES_DIR)
            if os.path.isfile(os.path.join(config.KNOWN_FACES_DIR, f)) and f.lower().endswith(valid_exts)
        ]
        for f in loose_files:
            name = os.path.splitext(f)[0].split("_")[0]  # e.g., 'Ashik_1.jpg' -> 'Ashik'
            img_path = os.path.join(config.KNOWN_FACES_DIR, f)
            img = load_image_robust(img_path)
            if img is None:
                continue
            
            vec = self.engine.extract_single_face_vector(img)
            if vec is not None:
                if name not in self.database:
                    self.database[name] = []
                self.database[name].append(vec)
                print(f"  + Encoded loose photo for '{name}'.")

        # Recompute centroids for any loose photos
        for name, vectors in self.database.items():
            if name not in self.centroids and vectors:
                avg_vec = np.mean(vectors, axis=0)
                norm_avg = avg_vec / max(1e-6, np.linalg.norm(avg_vec))
                self.centroids[name] = norm_avg

        # Save cache
        try:
            with open(config.DATABASE_CACHE_PATH, "wb") as f:
                pickle.dump({"database": self.database, "centroids": self.centroids}, f)
            print(f"\n[Face Database] Vector cache saved to: {config.DATABASE_CACHE_PATH}")
        except Exception as e:
            print(f"[Face Database] Could not write cache: {e}")

        print(f"\nTotal Enrolled Identities: {len(self.centroids)}")
        print("=" * 65 + "\n")

    def identify_face(self, live_feature: np.ndarray) -> Tuple[str, float]:
        """
        Matches a live 128-D face vector against enrolled centroids.
        Returns: (Name, Best Match Score 0.0 - 1.0)
        """
        if not self.centroids or live_feature is None:
            return "Unknown", 0.0

        best_name = "Unknown"
        best_score = -1.0

        for name, centroid_vec in self.centroids.items():
            score = self.engine.compute_cosine_similarity(live_feature, centroid_vec)
            if score > best_score:
                best_score = score
                best_name = name

        # Cosine similarity matching threshold
        if best_score >= config.COSINE_SIMILARITY_THRESHOLD:
            # Map raw SFace cosine score (typically 0.36 to 0.70) to human-friendly 0%-100% confidence
            norm_conf = min(0.99, max(0.50, (best_score - 0.20) / (0.75 - 0.20)))
            return best_name, norm_conf
        else:
            return "Unknown", max(0.0, float(best_score))

    def get_enrolled_names(self) -> List[str]:
        """Returns list of all enrolled person names."""
        return list(self.centroids.keys())
