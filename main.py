from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import shutil
from datetime import datetime
import sqlite3
from collections import defaultdict
from src.preprocessing import preprocess_data, vectorize_data, save_to_database, handle_class_imbalance, split_and_save_data, parse_resume, load_from_database
from src.model import evaluate_model, save_model
from src.prediction import predict_single, analyze_skills

app = FastAPI(
    title="ATS Match API",
    description="Machine Learning API for matching resumes to job descriptions",
    version="1.0",
    openapi_tags=[{
        "name": "predictions",
        "description": "Resume-job matching operations"
    }]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/best_optimized_model.keras'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
DATABASE_PATH = 'database/ats_database.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Global variables
model = load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
retraining_in_progress = False

MODEL_INFO = {
    "last_retrained": datetime.now().isoformat(),
    "performance": "Accuracy 82%, F1 Score 0.81",
    "data_size": "8000 samples"
}

# Sample data for testing
SAMPLE_RESUME = """
John Doe
Software Engineer
Skills: Python, FastAPI, TensorFlow, Machine Learning
Experience: 5 years at Tech Corp
Education: BS in Computer Science
"""

SAMPLE_JOB_DESC = """
Looking for a Machine Learning Engineer with:
- 3+ years Python experience
- Knowledge of TensorFlow/Keras
- API development skills
Competitive salary offered.
"""
@app.get("/", include_in_schema=False)
async def api_overview():
    """Returns overview of all API endpoints"""
    return {
        "message": "ATS Match API",
        "endpoints": {
            "GET /status": "Check API status",
            "POST /api/predict": "Predict match with text inputs (uses samples if empty)",
            "POST /api/predict_resume_file": "Predict match with resume file",
            "POST /api/upload": "Upload CSV dataset",
            "GET /api/retrain_status": "Check retraining status",
            "POST /api/retrain": "Trigger model retraining"
        },
        "try_it": "Visit /docs for interactive testing"
    }

@app.get("/status")
async def status():
    return {
          "status": "ready" if model and vectorizer else "not_ready",
        "last_retrained": MODEL_INFO["last_retrained"],
        "performance": MODEL_INFO["performance"],
        "data_size": MODEL_INFO["data_size"],
        "metrics": MODEL_INFO.get("metrics", {})
    }

@app.post("/api/predict")
async def predict(
    resume_text: str = Form(default=SAMPLE_RESUME),
    job_text: str = Form(default=SAMPLE_JOB_DESC)
):
    """
    Predict resume-job match probability
    
    Parameters:
    - resume_text: Resume content (default: sample resume)
    - job_text: Job description (default: sample job)
    
    Returns:
    - prediction: 1 (Relevant) or 0 (Not Relevant)
    - probability: Match confidence (0-1)
    """
    if not resume_text or not job_text:
        return JSONResponse(content={'error': 'Missing input text'}, status_code=400)
    
    prediction, probability = predict_single(resume_text, job_text, model, vectorizer)
    return {
        'prediction': int(prediction),
        'prediction_label': 'Relevant' if prediction == 1 else 'Not Relevant',
        'probability': float(probability),
        'note': 'Using sample data if no input provided'
    }

@app.post("/api/predict_resume_file")
async def predict_resume_file(resume: UploadFile = File(...), job_text: str = Form(...)):
    if not resume.filename.endswith((".pdf", ".docx", ".doc")):
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    # Save the uploaded file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    with open(temp_path, "wb") as f:
        content = await resume.read()
        f.write(content)

    # Parse resume into plain text
    try:
        resume_text = parse_resume(temp_path)
    except Exception as e:
        return JSONResponse(content={"error": f"Resume parsing failed: {str(e)}"}, status_code=500)

    # Predict
    prediction, probability, skills_info = predict_single(resume_text, job_text, model, vectorizer)
    
    return {
        "prediction": int(prediction),
        "prediction_label": "Relevant" if prediction == 1 else "Not Relevant",
        "probability": float(probability),
        "matching_skills": skills_info["matching_skills"],
        "missing_skills": skills_info["missing_skills"],
        "resume_excerpt": resume_text[:300]
    }

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return JSONResponse(content={"error": "Only CSV files are allowed"}, status_code=400)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = pd.read_csv(filepath)
    save_to_database(df, DATABASE_PATH)
    return {"filename": file.filename, "rows": len(df)}

@app.get("/api/retrain_status")
async def retrain_status():
    return {"retraining": retraining_in_progress}

def retrain_model_task():
    global model, vectorizer, retraining_in_progress, MODEL_INFO
    retraining_in_progress = True
    try:
        # Load data from database
        df = load_from_database(db_path=DATABASE_PATH)
        
        # Check if we have enough data
        if len(df) < 10:  # Set a minimum threshold
            raise ValueError(f"Not enough training data. Found only {len(df)} samples, need at least 10.")
        
        # Update data size
        MODEL_INFO["data_size"] = f"{len(df):,} samples"
        
        processed_df = preprocess_data(df)
        _, _ = split_and_save_data(processed_df)
        
        # Create new vectorizer and transform data
        X_features, y, new_vectorizer = vectorize_data(processed_df)
        
        # Check class distribution
        class_counts = {c: sum(y == c) for c in set(y)}
        print(f"Class distribution before balancing: {class_counts}")
        
        if len(class_counts) < 2:
            raise ValueError(f"Only one class found in the data: {class_counts}. Need both positive and negative examples.")
        
        # Handle class imbalance
        X_balanced, y_balanced = handle_class_imbalance(X_features, y, method='smote')
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
        
        # Get input shape for new model
        input_shape = X_train.shape[1]
        print(f"Building new model with input shape: {input_shape}")
        
        # Create a new model with the current input shape
        new_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the new model
        new_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-5)
        ]
        
        # Train the new model
        history = new_model.fit(
            X_train, y_train, 
            epochs=20, 
            batch_size=32, 
            validation_data=(X_test, y_test), 
            callbacks=callbacks
        )
        
        # Evaluate the new model
        eval_metrics = new_model.evaluate(X_test, y_test)
        loss = eval_metrics[0]
        accuracy = eval_metrics[1]
        
        # Get additional metrics
        y_pred = (new_model.predict(X_test) > 0.5).astype("int32")
        from sklearn.metrics import f1_score, recall_score
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Update model info
        MODEL_INFO["performance"] = f"Accuracy {accuracy:.1%}, F1 Score {f1:.2f}"
        MODEL_INFO["last_retrained"] = datetime.now().isoformat()
        MODEL_INFO["metrics"] = {
            "loss": float(loss),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "f1": float(f1)
        }
        
        # Replace the old model and vectorizer with the new ones
        model = new_model
        vectorizer = new_vectorizer
        
        # Save the new model and vectorizer
        save_model(model, MODEL_PATH)
        joblib.dump(new_vectorizer, VECTORIZER_PATH)
        
        print("Model retrained and saved successfully!")
        
    except Exception as e:
        print(f"Retraining error: {e}")
        import traceback
        traceback.print_exc()  # Print full error details
    finally:
        retraining_in_progress = False


def initialize_model_info():
    global MODEL_INFO
    try:
        # Try to get dataset size from database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM resume_job_matches")
        count = cursor.fetchone()[0]
        conn.close()
        
        MODEL_INFO["data_size"] = f"{count:,} samples"
        
        # Initialize metrics to ensure they're always available
        if "metrics" not in MODEL_INFO:
            MODEL_INFO["metrics"] = {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0
            }
    except Exception as e:
        print(f"Error getting initial data size: {e}")
        # Still initialize metrics even if database access fails
        if "metrics" not in MODEL_INFO:
            MODEL_INFO["metrics"] = {
                "loss": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "recall": 0.0
            }

@app.post("/api/retrain")
async def retrain():
    global retraining_in_progress
    if retraining_in_progress:
        return JSONResponse(content={"error": "Retraining already in progress"}, status_code=400)

    thread = threading.Thread(target=retrain_model_task)
    thread.start()
    return {"message": "Retraining started from database"}

# To run: uvicorn main:app --reload

