import cv2
import numpy as np

def draw_transparent_rect(img, x, y, w, h, color, alpha=0.35, border_color=None, border_thickness=2):
    """Draw a modern translucent rectangle on an image."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border_color is not None:
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, border_thickness)

def draw_corner_brackets(img, x, y, w, h, color=(0, 255, 0), thickness=2, length=15):
    """Draw tactical HUD corner brackets around a bounding box."""
    # Top-Left
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

def draw_header_hud(frame, title="AI ASSISTIVE VISION", subtitle="Walking Assistance Active", fps=0.0, is_muted=False):
    """Draw a clean, sleek header HUD at top of frame."""
    h, w = frame.shape[:2]
    # Header bar
    draw_transparent_rect(frame, 10, 10, w - 20, 60, (20, 20, 25), alpha=0.75, border_color=(70, 70, 80), border_thickness=1)

    # Title & Subtitle
    cv2.putText(frame, title, (25, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 255), 1, cv2.LINE_AA)

    # FPS counter
    fps_text = f"FPS: {fps:4.1f}"
    cv2.putText(frame, fps_text, (w - 180, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

    # Voice Status Pill
    voice_color = (0, 0, 255) if is_muted else (0, 220, 0)
    voice_text = "VOICE: MUTED (V)" if is_muted else "VOICE: ON (V)"
    cv2.putText(frame, voice_text, (w - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, voice_color, 1, cv2.LINE_AA)

def draw_navigation_zones(frame, left_boundary, right_boundary, ground_horizon_ratio=0.45, alpha=0.15):
    """Draw Left, Center, and Right navigation zone corridors on the floor."""
    h, w = frame.shape[:2]
    y_start = int(h * ground_horizon_ratio)

    overlay = frame.copy()
    
    # Left Zone
    cv2.rectangle(overlay, (0, y_start), (int(w * left_boundary), h), (255, 140, 0), -1)
    # Center Zone (Safe corridor focus)
    cv2.rectangle(overlay, (int(w * left_boundary), y_start), (int(w * right_boundary), h), (0, 200, 0), -1)
    # Right Zone
    cv2.rectangle(overlay, (int(w * right_boundary), y_start), (w, h), (255, 140, 0), -1)
    
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Zone Dividers
    cv2.line(frame, (int(w * left_boundary), y_start), (int(w * left_boundary), h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (int(w * right_boundary), y_start), (int(w * right_boundary), h), (255, 255, 255), 1, cv2.LINE_AA)
    
    # Labels positioned right beneath the horizon line
    label_y = y_start + 22
    cv2.putText(frame, "LEFT ZONE", (int(w * (left_boundary / 2)) - 35, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "MAIN PATH", (int(w * 0.5) - 35, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT ZONE", (int(w * (right_boundary + (1 - right_boundary) / 2)) - 40, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

def draw_guidance_banner(frame, command_text, direction_code="FORWARD", reason_text=""):
    """
    Draw bottom guidance instruction badge with colored alert state.
    direction_code: 'FORWARD', 'LEFT', 'RIGHT', 'STOP'
    """
    h, w = frame.shape[:2]
    banner_h = 75
    banner_y = h - banner_h - 15
    banner_w = w - 40
    banner_x = 20

    # Color mapping
    colors = {
        "FORWARD": (0, 160, 50),     # Vibrant Green
        "LEFT": (220, 140, 0),       # Orange / Amber
        "RIGHT": (220, 140, 0),      # Orange / Amber
        "STOP": (0, 0, 220),         # Urgent Red
        "CAUTION": (0, 180, 220)     # Yellow
    }
    bg_color = colors.get(direction_code, (100, 100, 100))

    # Banner box
    draw_transparent_rect(frame, banner_x, banner_y, banner_w, banner_h, bg_color, alpha=0.85, border_color=(255, 255, 255), border_thickness=2)

    # Command text
    cv2.putText(frame, command_text.upper(), (banner_x + 25, banner_y + 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Sub reason text
    if reason_text:
        cv2.putText(frame, reason_text, (banner_x + 25, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 245, 255), 1, cv2.LINE_AA)

    # Direction Arrow / Symbol Graphic on Right of Banner
    symbol_x = banner_x + banner_w - 50
    symbol_y = banner_y + int(banner_h / 2)
    
    if direction_code == "FORWARD":
        # Upward Arrow
        cv2.arrowedLine(frame, (symbol_x, symbol_y + 18), (symbol_x, symbol_y - 18), (255, 255, 255), 4, tipLength=0.4)
    elif direction_code == "LEFT":
        # Left Arrow
        cv2.arrowedLine(frame, (symbol_x + 18, symbol_y), (symbol_x - 18, symbol_y), (255, 255, 255), 4, tipLength=0.4)
    elif direction_code == "RIGHT":
        # Right Arrow
        cv2.arrowedLine(frame, (symbol_x - 18, symbol_y), (symbol_x + 18, symbol_y), (255, 255, 255), 4, tipLength=0.4)
    elif direction_code == "STOP":
        # Stop Hand / Hexagon
        cv2.circle(frame, (symbol_x, symbol_y), 20, (255, 255, 255), -1)
        cv2.putText(frame, "!", (symbol_x - 5, symbol_y + 8), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
