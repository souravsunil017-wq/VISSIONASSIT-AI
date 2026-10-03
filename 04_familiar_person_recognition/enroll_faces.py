"""
Standalone One-Click Face Enrollment Script.
Scans the 'known_faces/' folder, encodes all images, and saves 'face_database.pkl'.
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_database import FaceDatabase

def main():
    print("=" * 65)
    print("       FAMILIAR PERSON ENROLLMENT & ENCODING TOOL")
    print("=" * 65)
    db = FaceDatabase()
    db.rebuild_database()
    print("[SUCCESS] Enrollment process completed successfully!")
    print(f"Enrolled identities: {db.get_enrolled_names()}")

if __name__ == "__main__":
    main()
