from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banana Classification API",
    description="API for classifying banana ripeness using EfficientNet",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
img_size = (224, 224)

# Model configuration
MODEL_PATH = "model/banana_model_FINAL.keras"  # Adjust path as needed
MODEL_INFO = {
    "name": "Banana Classifier",
    "version": "1.0",
    "architecture": "EfficientNetB0",
    "classes": class_names,
    "input_size": img_size,
    "trained_on": "Custom banana dataset"
}


def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image exactly as done during training
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model input size
        image = image.resize(img_size)

        # Convert to numpy array
        img_array = np.array(image)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Convert to float32
        img_array = tf.cast(img_array, tf.float32)

        # Apply EfficientNet preprocessing (same as training)
        img_array = keras.applications.efficientnet.preprocess_input(img_array)

        return img_array

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Banana Classification API",
        "version": "1.0.0",
        "description": "Upload banana images to classify their ripeness",
        "endpoints": {
            "/": "This endpoint",
            "/health": "Health check",
            "/model/info": "Model information",
            "/predict": "Predict banana ripeness (POST with image)",
            "/predict/batch": "Batch prediction (POST with multiple images)",
            "/docs": "API documentation"
        },
        "usage": "POST image to /predict endpoint",
        "classes": class_names
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": model_status,
        "api_version": "1.0.0"
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get model summary info
        total_params = model.count_params()
        trainable_params = sum([tf.reduce_prod(var.shape) for var in model.trainable_variables])

        return {
            **MODEL_INFO,
            "model_loaded": True,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "model_size_mb": round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2) if os.path.exists(
                MODEL_PATH) else "unknown"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model information")


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """
    Predict banana ripeness for a single image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]

        # Get all class probabilities
        class_probabilities = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }

        return {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": confidence,
                "class_index": int(predicted_class_idx)
            },
            "all_probabilities": class_probabilities,
            "metadata": {
                "filename": file.filename,
                "file_size": len(image_data),
                "image_size": image.size,
                "processed_size": img_size,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict banana ripeness for multiple images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    results = []

    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "File must be an image"
            })
            continue

        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))

            # Preprocess image
            processed_image = preprocess_image(image)

            # Make prediction
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = class_names[predicted_class_idx]

            # Get all class probabilities
            class_probabilities = {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }

            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": {
                    "class": predicted_class,
                    "confidence": confidence,
                    "class_index": int(predicted_class_idx)
                },
                "all_probabilities": class_probabilities
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "processed_files": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/classes")
async def get_classes():
    """Get all available classes"""
    return {
        "classes": class_names,
        "num_classes": len(class_names),
        "descriptions": {
            "overripe": "Banana is past optimal ripeness",
            "ripe": "Banana is perfectly ripe and ready to eat",
            "rotten": "Banana is spoiled and not edible",
            "unripe": "Banana is not yet ripe"
        }
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn

    # Update MODEL_PATH if needed
    if not os.path.exists(MODEL_PATH):
        # Try alternative paths
        alternative_paths = [
            "banana_model_final.keras",
            "banana_model_FINAL.keras",
            "../model/banana_model_FINAL.keras"
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                MODEL_PATH = alt_path
                break

    print(f"Starting server...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")

    uvicorn.run(app, host="0.0.0.0", port=8000)