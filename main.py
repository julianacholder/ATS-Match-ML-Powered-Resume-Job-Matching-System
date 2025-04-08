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
from collections import defaultdict
# Add SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psutil

# Import prediction functions
from src.preprocessing import preprocess_data, vectorize_data, handle_class_imbalance, split_and_save_data, parse_resume
from src.model import evaluate_model, save_model
from src.prediction import predict_single, analyze_skills

from dotenv import load_dotenv
load_dotenv()


tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL', 'fallback-connection-string-for-development')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class ResumeJobMatch(Base):
    __tablename__ = "resume_job_matches"
    
    id = Column(Integer, primary_key=True, index=True)
    career_objective = Column(Text, nullable=True)
    skills = Column(Text, nullable=True)
    degree_names = Column(Text, nullable=True)
    positions = Column(Text, nullable=True)
    job_position_name = Column(String(255), nullable=True)
    job_description = Column(Text, nullable=True)
    match = Column(Integer, nullable=False)

class ModelInfo(Base):
    __tablename__ = "model_info"
    
    id = Column(Integer, primary_key=True, index=True)
    last_retrained = Column(String(50), nullable=False)
    performance = Column(String(255), nullable=False)
    data_size = Column(String(50), nullable=False)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Database functions
def save_to_database(df, db_path=None):
    """Save DataFrame to PostgreSQL database"""
    session = SessionLocal()
    
    try:
        for _, row in df.iterrows():
            db_item = ResumeJobMatch(
                career_objective=str(row.get('career_objective', '')),
                skills=str(row.get('skills', '')),
                degree_names=str(row.get('degree_names', '')),
                positions=str(row.get('positions', '')),
                job_position_name=str(row.get('job_position_name', '')),
                job_description=str(row.get('job_description', '')),
                match=int(row.get('match', 0))
            )
            session.add(db_item)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Database error: {e}")
        raise e
    finally:
        session.close()

def load_from_database(db_path=None):
    """Load data from PostgreSQL database into DataFrame"""
    import pandas as pd
    
    session = SessionLocal()
    
    try:
        # Query all records
        items = session.query(ResumeJobMatch).all()
        
        # Convert to DataFrame
        data = []
        for item in items:
            data.append({
                'career_objective': item.career_objective,
                'skills': item.skills,
                'degree_names': item.degree_names,
                'positions': item.positions,
                'job_position_name': item.job_position_name,
                'job_description': item.job_description,
                'match': item.match
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading from database: {e}")
        # Return empty DataFrame in case of error
        return pd.DataFrame()
    finally:
        session.close()

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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
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

@app.get("/memory_status")
def memory_status():
    mem = psutil.virtual_memory()
    return {
        "used_mb": mem.used / (1024 ** 2),
        "available_mb": mem.available / (1024 ** 2),
    }

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
    save_to_database(df)
    return {"filename": file.filename, "rows": len(df)}

@app.get("/api/retrain_status")
async def retrain_status():
    stage = "preprocessing"
    if hasattr(retrain_model_task, 'current_stage'):
        stage = retrain_model_task.current_stage
    return {"retraining": retraining_in_progress, "stage": stage}

def initialize_model_info():
    global MODEL_INFO
    
    session = SessionLocal()
    try:
        # Check if model info exists in database
        model_info = session.query(ModelInfo).first()
        
        if model_info:
            # Load from database
            MODEL_INFO = {
                "last_retrained": model_info.last_retrained,
                "performance": model_info.performance,
                "data_size": model_info.data_size,
                "metrics": {
                    "loss": model_info.loss or 0.0,
                    "accuracy": model_info.accuracy or 0.0,
                    "f1": model_info.f1 or 0.0,
                    "recall": model_info.recall or 0.0
                }
            }
        else:
            # Initialize with defaults and save to database
            MODEL_INFO = {
                "last_retrained": datetime.now().isoformat(),
                "performance": "Accuracy 82%, F1 Score 0.81",
                "data_size": "8000 samples",
                "metrics": {
                    "loss": 0.4326,
                    "accuracy": 0.82,
                    "f1": 0.81,
                    "recall": 0.8132
                }
            }
            
            # Count samples in database
            count = session.query(ResumeJobMatch).count()
            if count > 0:
                MODEL_INFO["data_size"] = f"{count:,} samples"
            
            # Save to database
            new_info = ModelInfo(
                last_retrained=MODEL_INFO["last_retrained"],
                performance=MODEL_INFO["performance"],
                data_size=MODEL_INFO["data_size"],
                loss=MODEL_INFO["metrics"]["loss"],
                accuracy=MODEL_INFO["metrics"]["accuracy"],
                f1=MODEL_INFO["metrics"]["f1"],
                recall=MODEL_INFO["metrics"]["recall"]
            )
            session.add(new_info)
            session.commit()
            
    except Exception as e:
        print(f"Error initializing model info: {e}")
    finally:
        session.close()

def retrain_model_task():
    global model, vectorizer, retraining_in_progress, MODEL_INFO
    retraining_in_progress = True
    retrain_model_task.current_stage = "preprocessing"
    
    try:
        # Load data from database
        df = load_from_database()
        
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
        
        # Update stage
        retrain_model_task.current_stage = "training"
        
        # Get input shape for new model
        input_shape = X_train.shape[1]
        print(f"Building new model with input shape: {input_shape}")
        
        # Create a new model with the current input shape
        new_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
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
        
        # Update stage
        retrain_model_task.current_stage = "evaluation"
        
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
        
        # Save model info to database
        session = SessionLocal()
        try:
            # Get existing record or create new
            model_info = session.query(ModelInfo).first()
            if not model_info:
                model_info = ModelInfo()
                
            # Update fields
            model_info.last_retrained = MODEL_INFO["last_retrained"]
            model_info.performance = MODEL_INFO["performance"]
            model_info.data_size = MODEL_INFO["data_size"]
            model_info.loss = MODEL_INFO["metrics"]["loss"]
            model_info.accuracy = MODEL_INFO["metrics"]["accuracy"]
            model_info.f1 = MODEL_INFO["metrics"]["f1"]
            model_info.recall = MODEL_INFO["metrics"]["recall"]
            
            session.add(model_info)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error saving model info: {e}")
        finally:
            session.close()
        
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

@app.post("/api/retrain")
async def retrain():
    global retraining_in_progress
    if retraining_in_progress:
        return JSONResponse(content={"error": "Retraining already in progress"}, status_code=400)

    thread = threading.Thread(target=retrain_model_task)
    thread.start()
    return {"message": "Retraining started from database"}

# Initialize on startup
initialize_model_info()

# Add a debug endpoint
@app.get("/debug")
async def debug_info():
    """Returns debug information about the API state"""
    return {
        "model_info": MODEL_INFO,
        "paths": {
            "upload_folder": os.path.abspath(UPLOAD_FOLDER),
            "model_path": os.path.abspath(MODEL_PATH),
        },
        "exists": {
            "model": os.path.exists(MODEL_PATH),
            "vectorizer": os.path.exists(VECTORIZER_PATH),
        },
        "database": {
            "url": DATABASE_URL.replace(DATABASE_URL.split("@")[0], "postgresql://****:****"),
            "resume_count": SessionLocal().query(ResumeJobMatch).count(),
            "model_info_exists": SessionLocal().query(ModelInfo).first() is not None
        },
        "environment": {k: v for k, v in os.environ.items() if "key" not in k.lower() and "secret" not in k.lower() and "password" not in k.lower()},
        "working_dir": os.getcwd(),
    }

# To run: uvicorn main:app --reload