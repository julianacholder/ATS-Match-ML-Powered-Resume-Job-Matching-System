from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import os
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import shutil
from src.preprocessing import preprocess_data, vectorize_data, save_to_database, handle_class_imbalance, split_and_save_data, parse_resume, load_from_database
from src.model import evaluate_model, save_model
from src.prediction import predict_single

app = FastAPI()

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

@app.get("/status")
async def status():
    return {"status": "ready" if model and vectorizer else "not_ready"}

@app.post("/api/predict")
async def predict(resume_text: str = Form(...), job_text: str = Form(...)):
    if not resume_text or not job_text:
        return JSONResponse(content={'error': 'Missing input text'}, status_code=400)
    prediction, probability = predict_single(resume_text, job_text, model, vectorizer)
    return {
        'prediction': int(prediction),
        'prediction_label': 'Relevant' if prediction == 1 else 'Not Relevant',
        'probability': float(probability)
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
    prediction, probability = predict_single(resume_text, job_text, model, vectorizer)
    return {
        "prediction": int(prediction),
        "prediction_label": "Relevant" if prediction == 1 else "Not Relevant",
        "probability": float(probability),
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
    global model, vectorizer, retraining_in_progress
    retraining_in_progress = True
    try:
        df = load_from_database(db_path=DATABASE_PATH)
        processed_df = preprocess_data(df)
        _, _ = split_and_save_data(processed_df)
        X_features, y, new_vectorizer = vectorize_data(processed_df)
        X_balanced, y_balanced = handle_class_imbalance(X_features, y, method='smote')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                      loss='binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, min_lr=1e-5)
        ]

        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)
        save_model(model, MODEL_PATH)
        joblib.dump(new_vectorizer, VECTORIZER_PATH)
        vectorizer = new_vectorizer

    except Exception as e:
        print(f"Retraining error: {e}")
    finally:
        retraining_in_progress = False

@app.post("/api/retrain")
async def retrain():
    global retraining_in_progress
    if retraining_in_progress:
        return JSONResponse(content={"error": "Retraining already in progress"}, status_code=400)

    thread = threading.Thread(target=retrain_model_task)
    thread.start()
    return {"message": "Retraining started from database"}

# To run: uvicorn main:app --reload

