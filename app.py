from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from pose_estimation import get_pose_keypoints_and_annotated_image
import logging
import traceback
from typing import Optional
import cv2
import numpy as np
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pose Estimation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Validate and process the uploaded image"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        # Check image dimensions
        height, width = image.shape[:2]
        if width < 100 or height < 100:
            raise ValueError("Image dimensions too small")
            
        # Check image format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image format")
            
        return image
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return None

@app.post("/analyze-pose")
async def analyze_pose(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Validate image
        image = validate_image(contents)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format or corrupted image"
            )
            
        # Process the image
        try:
            keypoints, pose_name, annotated_image_bytes = get_pose_keypoints_and_annotated_image(contents)
        except Exception as e:
            logger.error(f"Pose estimation error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error in pose estimation: {str(e)}"
            )
        
        # Convert annotated image to base64
        try:
            annotated_image_base64 = base64.b64encode(annotated_image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Base64 encoding error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error encoding processed image"
            )
        
        return {
            "status": "success",
            "pose": pose_name,
            "keypoints": keypoints,
            "annotated_image": annotated_image_base64
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
