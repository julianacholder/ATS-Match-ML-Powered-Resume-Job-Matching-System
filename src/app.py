from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
import sqlite3
from werkzeug.utils import secure_filename

# Import functions from our modules
from src.preprocessing import preprocess_data, vectorize_data, save_to_database, parse_resume, handle_class_imbalance, split_and_save_data
from src.model import train_model, evaluate_model, save_model
from src.prediction import predict_single, predict_batch

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'docx', 'doc'}
MODEL_PATH = 'models/best_model.keras'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
DATABASE_PATH = 'data/ats_database.db'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store model and vectorizer
model = None
vectorizer = None
retraining_in_progress = False

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_artifacts():
    """Load model and vectorizer"""
    global model, vectorizer
    
    try:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        
        print("Loading vectorizer...")
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        print("Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False
    
    return True

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/resume_match', methods=['GET', 'POST'])
def resume_match():
    """Match a resume with a job description"""
    if request.method == 'POST':
        # Check if files were uploaded
        if 'resume' not in request.files:
            return render_template('resume_match.html', error="No resume file uploaded")
        
        resume_file = request.files['resume']
        job_description = request.form.get('job_description', '')
        
        # Check if resume file was selected
        if resume_file.filename == '':
            return render_template('resume_match.html', error="No resume file selected")
        
        # Check file extension
        allowed_extensions = {'pdf', 'docx', 'doc'}
        if not '.' in resume_file.filename or \
           resume_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template('resume_match.html', 
                                  error="Invalid file format. Please upload PDF or Word document")
        
        try:
            # Save the resume file temporarily
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                      secure_filename(resume_file.filename))
            resume_file.save(resume_path)
            
            # Parse the resume
            resume_text = parse_resume(resume_path)
            
            # Make prediction
            prediction, probability = predict_single(
                resume_text=resume_text,
                job_text=job_description,
                model=model,
                vectorizer=vectorizer
            )
            
            # Get training data statistics
            stats = get_training_data_stats()
            
            # Return result
            result = {
                'prediction': 'Relevant' if prediction == 1 else 'Not Relevant',
                'probability': float(probability),
                'resume_text': resume_text[:500] + '...' if len(resume_text) > 500 else resume_text
            }
            
            return render_template('resume_match.html', result=result, job_description=job_description, stats=stats)
        
        except Exception as e:
            return render_template('resume_match.html', error=f"Error processing files: {e}")
    
    # GET request - just show the form with statistics
    stats = get_training_data_stats()
    return render_template('resume_match.html', stats=stats)

@app.route('/predict_single', methods=['GET', 'POST'])
def predict_single_endpoint():
    """Make a prediction for a single resume-job pair"""
    if request.method == 'POST':
        # Get data from form
        resume_text = request.form.get('resume_text', '')
        job_text = request.form.get('job_text', '')
        
        # Check if inputs are provided
        if not resume_text or not job_text:
            return render_template('predict.html', error="Please provide both resume and job description")
        
        try:
            # Make prediction
            prediction, probability = predict_single(
                resume_text=resume_text,
                job_text=job_text,
                model=model,
                vectorizer=vectorizer
            )
            
            # Return result
            result = {
                'prediction': 'Relevant' if prediction == 1 else 'Not Relevant',
                'probability': float(probability)
            }
            
            return render_template('predict.html', result=result, resume_text=resume_text, job_text=job_text)
        
        except Exception as e:
            return render_template('predict.html', error=f"Error making prediction: {e}")
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for prediction"""
    try:
        # Get data from request
        data = request.json
        resume_text = data.get('resume_text', '')
        job_text = data.get('job_text', '')
        
        # Check if inputs are provided
        if not resume_text or not job_text:
            return jsonify({'error': 'Please provide both resume and job description'}), 400
        
        # Make prediction
        prediction, probability = predict_single(
            resume_text=resume_text,
            job_text=job_text,
            model=model,
            vectorizer=vectorizer
        )
        
        # Return result
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': 'Relevant' if prediction == 1 else 'Not Relevant',
            'probability': float(probability)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Upload a CSV file for batch prediction or retraining"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', error="No file uploaded")
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('upload.html', error="No file selected")
        
        # Check if file has an allowed extension
        if not allowed_file(file.filename):
            return render_template('upload.html', error="File type not allowed. Please upload a CSV file")
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load file
            df = pd.read_csv(filepath)
            
            # Save to database
            save_to_database(df, DATABASE_PATH)
            
            return render_template('upload.html', success=f"File '{filename}' uploaded successfully", df_shape=df.shape)
        
        except Exception as e:
            return render_template('upload.html', error=f"Error processing file: {e}")
    
    return render_template('upload.html')

def get_training_data_stats():
    """Get statistics about the training data"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total samples
        try:
            cursor.execute("SELECT COUNT(*) FROM training_data")
            total_samples = cursor.fetchone()[0]
            
            # Get relevant matches
            cursor.execute("SELECT COUNT(*) FROM training_data WHERE prediction = 1")
            relevant_matches = cursor.fetchone()[0]
            
            # Get non-relevant matches
            cursor.execute("SELECT COUNT(*) FROM training_data WHERE prediction = 0")
            non_relevant_matches = cursor.fetchone()[0]
        except:
            total_samples = 0
            relevant_matches = 0
            non_relevant_matches = 0
        
        conn.close()
        
        return {
            'total_samples': total_samples,
            'relevant_matches': relevant_matches,
            'non_relevant_matches': non_relevant_matches
        }
    except Exception as e:
        print(f"Error getting training data stats: {e}")
        return None

def retrain_model_task(filepath):
    """Retrain model in a separate thread"""
    global model, vectorizer, retraining_in_progress
    
    try:
        # Set flag
        retraining_in_progress = True
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Split and save data to train/test folders
        train_path, test_path = split_and_save_data(processed_df)
        
        # Vectorize data
        X_features, y, new_vectorizer = vectorize_data(processed_df)
        
        # Handle class imbalance
        X_balanced, y_balanced = handle_class_imbalance(X_features, y, method='smote')
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
        
        # Use the existing model for fine-tuning instead of creating a new one
        # First make sure we have a model loaded
        if model is None:
            print("Loading existing model for retraining...")
            model = load_model(MODEL_PATH)
        
        print("Fine-tuning the existing model with new data...")
        
        # Compile the existing model with a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model with existing model as starting point
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=3,
            min_lr=1e-5
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=20,  # Fewer epochs for fine-tuning
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save artifacts
        save_model(model, MODEL_PATH)
        joblib.dump(new_vectorizer, VECTORIZER_PATH)
        
        # Update global variables
        vectorizer = new_vectorizer
        
        print("Model retrained successfully")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error retraining model: {e}")
    
    finally:
        # Clear flag
        retraining_in_progress = False

@app.route('/retrain', methods=['GET', 'POST'])
def retrain_model_endpoint():
    """Trigger model retraining"""
    global retraining_in_progress
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('retrain.html', error="No file uploaded")
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return render_template('retrain.html', error="No file selected")
        
        # Check if file has an allowed extension
        if not allowed_file(file.filename):
            return render_template('retrain.html', error="File type not allowed. Please upload a CSV file")
        
        # Check if retraining is already in progress
        if retraining_in_progress:
            return render_template('retrain.html', error="Retraining is already in progress")
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Start retraining in a separate thread
            thread = threading.Thread(target=retrain_model_task, args=(filepath,))
            thread.daemon = True
            thread.start()
            
            return render_template('retrain.html', success="Retraining started")
        
        except Exception as e:
            return render_template('retrain.html', error=f"Error starting retraining: {e}")
    
    return render_template('retrain.html')

@app.route('/status')
def status():
    """Check if model and vectorizer are loaded"""
    if model is not None and vectorizer is not None:
        return jsonify({'status': 'ready'})
    else:
        return jsonify({'status': 'not_ready'})

@app.route('/retrain_status')
def retrain_status():
    """Check retraining status"""
    return jsonify({'retraining': retraining_in_progress})

@app.route('/visualize')
def visualize():
    """Visualization page"""
    return render_template('visualize.html')

if __name__ == '__main__':
    # Load model and vectorizer
    load_artifacts()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)