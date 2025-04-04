# 🧠 ATS-Match: ML-Powered Resume-Job Matching System

##  Project Overview

**ATS-Match** is an AI-powered Applicant Tracking System that predicts the relevance of resumes for specific job postings using machine learning. The system combines natural language processing and neural networks to help job seekers understand how well their resume matches with job requirements, and assists recruiters in finding the most suitable candidates.

---

##  Quick Links

- ** Live Demo** – https://ats-frontend-delta.vercel.app 
- ** API Docs** – https://ats-match-ml-powered-resume-job-production.up.railway.app/docs 
- ** Video Demo** – Complete walkthrough of features and deployment  

---

##  Features

-  **Resume-Job Matching**: Predicts relevance scores between resumes and job descriptions  
-  **Data Visualization**: Interactive charts showing key factors in resume-job matching  
-  **Database Integration**: SQLite for storing and retrieving training data  
-  **Model Retraining**: Upload new data and trigger model retraining  
-  **Cloud Deployment**: Fully deployed with Docker, Render & Vercel

---

##  Technology Stack

| Area            | Tools Used                          |
|-----------------|--------------------------------------|
| **Frontend**    | HTML, CSS, JavaScript               |
| **Backend**     | FastAPI                      |
| **ML**          | TensorFlow, scikit-learn            |
| **Data**        | Pandas, NumPy                       |
| **Visualization** | Matplotlib, Seaborn               |
| **Deployment**  | Docker, Vercel, Render              |

---

##  Model Performance

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 82%     |
| Precision  | 79.7%     |
| Recall     | 83.2%     |
| F1 Score   | 81.3%     |

---

## 📁 Project Structure

```
ATS-Match/
│
├── README.md               # Project documentation
├── notebook/
│   └── ATS_Deployment.ipynb     # Jupyter Notebook for pipeline demo
│
├── src/
│   ├── preprocessing.py         # Data loading and preprocessing
│   ├── model.py                 # Model training and evaluation
│   └── prediction.py            # Inference and prediction
│
├── data/
│   ├── train/                   # Training data
│   └── test/                    # Test data
│
├── models/
│   ├── best_model.keras         # Pre-trained model
│   └── tfidf_vectorizer.pkl 
    └── ats_retrained_model.keras    

├── main.py                       # Fast Apu entry point
├── Dockerfile                   # Docker config
├── requirements.txt             # Project dependencies
              
```

---

##  Setup Instructions

###  Prerequisites

- Python 3.9+  
- pip  
- Docker (optional for containerized deployment)

###  Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/ats-match.git
cd ats-match

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py                   # uvicorn main:app --reload for FastAPI
```

Visit: [http://localhost:5000](http://localhost:5000)

---

###  Docker Deployment

```bash
# Build Docker image
docker build -t ats-match .

# Run container
docker run -p 5000:5000 ats-match
```

Visit: [http://localhost:5000](http://localhost:5000)

---

##  Visualizations and Insights

1. **Experience Level Impact** – How experience influences matching  
2. **Skill Match Distribution** – Relationship between skills and relevance  
3. **TF-IDF Feature Importance** – Key words driving match predictions  

---

##  ML Pipeline

1. **Data Preprocessing** – Text cleaning & feature engineering  
2. **Vectorization** – TF-IDF or BERT embeddings  
3. **Model Training** – Neural network with SGD  
4. **Evaluation** – Accuracy, Precision, Recall, F1  
5. **Deployment** – API with live retraining support  

---

##  API Documentation

### 🔹 Single Prediction Endpoint

- **[URL](https://ats-api-bywt.onrender.com)**: `/api/predict`  
- **Method**: `POST`

####  Request Body

```Form-data

  "resume_text": "Your resume text here",
  "job_text": "Job description text here"

```

#### 📥 Response

```json
{
  "prediction": 1,
  "prediction_label": "Relevant",
  "probability": 0.92
}
```

---

##  Contributing

Contributions are welcome!  
Feel free to open issues or submit Pull Requests 🙌

---

## 📄 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

##  Contact

**Your Name** – [j.holder@alustudent.com](j.holder@alustudent.com)  
GitHub Repo: [github.com/julianacholder/ATS-Match-ML-Powered-Resume-Job-Matching-System](https://github.com/julianacholder/ATS-Match-ML-Powered-Resume-Job-Matching-System)
