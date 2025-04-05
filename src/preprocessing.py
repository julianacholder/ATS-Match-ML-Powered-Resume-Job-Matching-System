import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(file_path):
    """
    Load data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a small sample dataset if file not found
        return create_sample_dataset()

def create_sample_dataset():
    """Sample dataset for demonstration purposes"""
    data = {
        'career_objective': ['Seeking a software engineering position', 'Looking for data science opportunities'],
        'skills': ['Python, Java, SQL', 'Python, R, Machine Learning'],
        'degree_names': ['BS Computer Science', 'MS Data Science'],
        'positions': ['Software Developer', 'Data Analyst'],
        'responsibilities': ['Developed web applications', 'Analyzed customer data'],
        '\ufeffjob_position_name': ['Software Engineer', 'Data Scientist'],
        'educationaL_requirements': ['Bachelor in CS or related', 'Masters in DS or related'],
        'experiencere_requirement': ['3+ years experience', '2+ years experience'],
        'responsibilities.1': ['Building software solutions', 'Developing ML models'],
        'skills_required': ['Java, Python, Git', 'Python, TensorFlow, SQL'],
        'matched_score': [0.85, 0.65]
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """
    Preprocess the data for the ATS model
    
    Args:
        df: DataFrame containing the raw data
        
    Returns:
        DataFrame with preprocessed features
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check for required columns and add them if missing
    required_columns = ['career_objective', 'skills', 'degree_names', 'positions', 
                       'responsibilities', '\ufeffjob_position_name', 'educationaL_requirements',
                       'experiencere_requirement', 'responsibilities.1', 'skills_required', 
                       'matched_score']
    
    for col in required_columns:
        if col not in df_processed.columns:
            print(f"Adding missing column: {col}")
            df_processed[col] = ''
    
    # Fix for job position name column
    if '\ufeffjob_position_name' not in df_processed.columns and 'job_position_name' in df_processed.columns:
        df_processed['\ufeffjob_position_name'] = df_processed['job_position_name']
    
    # Combine resume fields into 'resume_text'
    df_processed['resume_text'] = (
        df_processed['career_objective'].fillna('') + " " +
        df_processed['skills'].fillna('') + " " +
        df_processed['degree_names'].fillna('') + " " +
        df_processed['positions'].fillna('') + " " +
        df_processed['responsibilities'].fillna('')
    )
    
    # Combine job fields into 'job_text'
    job_position_col = '\ufeffjob_position_name' if '\ufeffjob_position_name' in df_processed.columns else 'job_position_name'
    
    df_processed['job_text'] = (
        df_processed[job_position_col].fillna('') + " " +
        df_processed['educationaL_requirements'].fillna('') + " " +
        df_processed['experiencere_requirement'].fillna('') + " " +
        df_processed['responsibilities.1'].fillna('') + " " +
        df_processed['skills_required'].fillna('')
    )
    
    # Combine the resume and job texts for classification
    df_processed['combined'] = df_processed['resume_text'] + " [SEP] " + df_processed['job_text']
    
    # Create the binary label (relevant if matched_score >= 0.7)
    # Handle case where matched_score might be renamed to 'match'
    if 'matched_score' in df_processed.columns:
        df_processed['job_match'] = (df_processed['matched_score'].astype(float) >= 0.7).astype(int)
    elif 'match' in df_processed.columns:
        df_processed['job_match'] = df_processed['match'].astype(int)
    else:
        # Default to 1 if no match column is found
        print("No match column found. Setting all matches to 1.")
        df_processed['job_match'] = 1
    
    return df_processed

def vectorize_data(df, vectorizer_path=None):
    """
    Convert text data to numerical features using TF-IDF
    
    Args:
        df: Preprocessed DataFrame
        vectorizer_path: Optional path to a saved vectorizer
        
    Returns:
        X_features: Feature matrix
        y: Target labels
        vectorizer: Fitted TF-IDF vectorizer
    """
    if vectorizer_path and os.path.exists(vectorizer_path):
        # Load saved vectorizer
        vectorizer = joblib.load(vectorizer_path)
        X_features = vectorizer.transform(df['combined']).toarray()
    else:
        # Create and fit a new vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X_features = vectorizer.fit_transform(df['combined']).toarray()
    
    y = df['job_match'].values
    
    return X_features, y, vectorizer

def handle_class_imbalance(X, y, method='smote'):
    """
    Handle class imbalance in the dataset
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Method to use ('smote', 'undersample', 'oversample')
        
    Returns:
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target labels
    """
    # Check if there is an imbalance
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if len(unique) < 2:
        print(f"Warning: Only one class found in the data: {class_counts}")
        return X, y
    
    imbalance_ratio = max(counts) / min(counts)
    print(f"Class counts: {class_counts}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # If ratio is below 1.5, consider it balanced enough
    if imbalance_ratio < 1.5:
        print("Class distribution is relatively balanced. Skipping balancing.")
        return X, y
    
    # Apply the specified method
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            print(f"Applied SMOTE: {dict(zip(unique_bal, counts_bal))}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            # Fall back to random oversampling
            method = 'oversample'
    
    if method == 'oversample':
        try:
            from imblearn.over_sampling import RandomOverSampler
            oversampler = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = oversampler.fit_resample(X, y)
            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            print(f"Applied oversampling: {dict(zip(unique_bal, counts_bal))}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Error applying oversampling: {e}")
            return X, y
    
    if method == 'undersample':
        try:
            from imblearn.under_sampling import RandomUnderSampler
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            print(f"Applied undersampling: {dict(zip(unique_bal, counts_bal))}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"Error applying undersampling: {e}")
            return X, y
    
    # If no method worked or specified
    return X, y

def parse_resume(file_path):
    """
    Extract text from a resume file (PDF or DOCX)
    
    Args:
        file_path: Path to the resume file
        
    Returns:
        Extracted text from the resume
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return parse_pdf(file_path)
    elif file_extension in ['docx', 'doc']:
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def parse_pdf(file_path):
    """Extract text from PDF file"""
    from pdfminer.high_level import extract_text
    
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        # Fallback to PyPDF2 if pdfminer fails
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e2:
            print(f"Error with fallback PDF extraction: {e2}")
            return ""

def parse_docx(file_path):
    """Extract text from DOCX file"""
    import docx
    
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def split_and_save_data(df, train_ratio=0.8, random_state=42):
    """
    Split data into train and test sets and save to respective folders
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion of data to use for training
        random_state: Random seed for reproducibility
        
    Returns:
        train_path: Path to saved training data
        test_path: Path to saved test data
    """
    # Create directories if they don't exist
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split data
    train_size = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled[:train_size]
    test_df = df_shuffled[train_size:]
    
    # Save data
    train_path = 'data/train/train_data.csv'
    test_path = 'data/test/test_data.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data saved to {train_path} ({len(train_df)} samples)")
    print(f"Test data saved to {test_path} ({len(test_df)} samples)")
    
    return train_path, test_path

def save_to_database(df, db_path='data/ats_database.db', table_name='resume_job_matches'):
    """
    Save DataFrame to a SQLite database
    
    Args:
        df: DataFrame to save
        db_path: Path to the database file
        table_name: Name of the table
        
    Returns:
        True if successful
    """
    import sqlite3
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Save DataFrame to database
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    # Close connection
    conn.close()
    
    return True

def load_from_database(db_path='data/ats_database.db', table_name='resume_job_matches'):
    """
    Load data from a SQLite database
    
    Args:
        db_path: Path to the database file
        table_name: Name of the table
        
    Returns:
        DataFrame containing the data
    """
    import sqlite3
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Load data from database
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, conn)
        
        # Close connection
        conn.close()
        
        return df
    
    except Exception as e:
        print(f"Error loading from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame instead of None