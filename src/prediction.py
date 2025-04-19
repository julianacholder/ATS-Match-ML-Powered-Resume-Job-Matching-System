import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import io
from contextlib import redirect_stdout
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Create SkillExtractor instance with default skill DB
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


def get_model_summary(model):
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    return stream.getvalue()

def preprocess_input(resume_text, job_text, vectorizer):
    """
    Preprocess a resume and job description for prediction
    
    Args:
        resume_text: Text of the resume
        job_text: Text of the job description
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        Preprocessed feature vector
    """
    # Combine resume and job text
    combined_text = resume_text + " [SEP] " + job_text
    
    # Vectorize the text
    features = vectorizer.transform([combined_text]).toarray()
    
    return features

def predict_single(resume_text, job_text, model, vectorizer):
    """
    Make a prediction for a single resume-job pair, combining model confidence and skill overlap.
    """
    features = preprocess_input(resume_text, job_text, vectorizer)
    probability = model.predict(features)[0][0]

    skills_info = analyze_skills(resume_text, job_text)
    matching_skills = skills_info["matching_skills"]

    # Compute combined score (optional)
    combined_score = (probability * 0.7) + (min(len(matching_skills), 5) / 10 * 0.3)

    # RULE OVERRIDE: If at least 5 matching skills, force it to Relevant
    if len(matching_skills) >= 5:
        prediction = 1
        reason = "Rule override: 5+ matching skills"
    else:
        prediction = 1 if combined_score >= 0.7 else 0
        reason = "Based on combined score and model confidence"

    return prediction, probability, skills_info, combined_score, reason



def extract_skills(text, min_score=0.7):
    """
    Extracts skills using both full matches and fuzzy matches.
    Returns a set of unique skill strings.
    """
    try:
        annotations = skill_extractor.annotate(text)

        # Exact matches
        full_matches = {
            skill['doc_node_value'].lower()
            for skill in annotations['results']['full_matches']
        }

        # Fuzzy matches above threshold
        fuzzy_matches = {
            skill['doc_node_value'].lower()
            for skill in annotations['results']['ngram_scored']
            if skill['score'] >= min_score
        }

        # Combine both sets
        return full_matches.union(fuzzy_matches)

    except Exception as e:
        print(f"Skill extraction error: {e}")
        return set()



def analyze_skills(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matching_skills = sorted(resume_skills & job_skills)
    missing_skills = sorted(job_skills - resume_skills)

    return {
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
    }


def predict_batch(resume_texts, job_texts, model, vectorizer):
    """
    Make predictions for multiple resume-job pairs
    
    Args:
        resume_texts: List of resume texts
        job_texts: List of job description texts
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        predictions: List of predictions (1 or 0)
        probabilities: List of confidence scores
    """
    # Check if inputs have the same length
    if len(resume_texts) != len(job_texts):
        raise ValueError("Number of resumes and jobs must be the same")
    
    # Preprocess each pair
    features_list = []
    for resume, job in zip(resume_texts, job_texts):
        features = preprocess_input(resume, job, vectorizer)
        features_list.append(features)
    
    # Combine all features
    batch_features = np.vstack(features_list)
    
    # Make predictions
    batch_probabilities = model.predict(batch_features).flatten()
    batch_predictions = (batch_probabilities >= 0.5).astype(int)
    
    return batch_predictions, batch_probabilities

def load_prediction_artifacts(model_path="models/best_model.keras", 
                             vectorizer_path="models/tfidf_vectorizer.pkl"):
    """
    Load model and vectorizer for prediction
    
    Args:
        model_path: Path to the model file
        vectorizer_path: Path to the vectorizer file
        
    Returns:
        model: Loaded model
        vectorizer: Loaded vectorizer
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Load vectorizer
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    
    except Exception as e:
        print(f"Error loading prediction artifacts: {e}")
        return None, None

def generate_detailed_report(resume_text, job_text, prediction, probability, vectorizer):
    """
    Generate a detailed report explaining the prediction
    
    Args:
        resume_text: Text of the resume
        job_text: Text of the job description
        prediction: Model prediction (1 or 0)
        probability: Confidence score
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        report: Dictionary containing the detailed report
    """
    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Preprocess the input
    combined_text = resume_text + " [SEP] " + job_text
    features = vectorizer.transform([combined_text]).toarray()[0]
    
    # Get the top contributing features
    top_indices = np.argsort(features)[-10:][::-1]
    top_features = [(feature_names[i], features[i]) for i in top_indices if features[i] > 0]
    
    # Check for key skills in both resume and job
    job_skills = [word.lower() for word in job_text.split() if len(word) > 3]
    resume_skills = [word.lower() for word in resume_text.split() if len(word) > 3]
    matching_skills = set(job_skills).intersection(set(resume_skills))
    
    # Generate the report
    report = {
        'prediction': 'Relevant' if prediction == 1 else 'Not Relevant',
        'confidence': probability,
        'top_features': top_features,
        'matching_skills': list(matching_skills)
    }
    
    return report