import cv2
import numpy as np
from io import BytesIO

def read_image_from_bytes(image_bytes):
    """Convert image bytes to OpenCV format"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Error reading image: {str(e)}")

def image_to_bytes(image):
    """Convert OpenCV image to bytes"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    except Exception as e:
        raise ValueError(f"Error converting image to bytes: {str(e)}")