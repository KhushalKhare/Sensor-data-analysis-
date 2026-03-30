# Health Sensor Data Analysis System

An end-to-end machine learning system for analyzing wearable sensor data and detecting anomalous physiological patterns in near real-time.

##  Problem Statement

Wearable devices generate continuous streams of physiological data (heart rate, activity levels, etc.), but extracting meaningful insights from this data is non-trivial due to:

- High volume and noise
- Lack of labeled anomalies
- Temporal variability in human behavior

This project addresses these challenges by building a scalable anomaly detection pipeline for health sensor data.

---

## 🚀 Solution Overview

This system implements an end-to-end pipeline:

1. **Data Ingestion & Processing**
   - Structured sensor data ingestion
   - Feature engineering and normalization

2. **Modeling**
   - Unsupervised anomaly detection using Isolation Forest
   - Detection of abnormal physiological patterns without labeled data

3. **Backend System**
   - Model loading and inference layer
   - Modular structure for scalability

4. **Frontend Interface**
   - Interactive visualization using Streamlit
   - Real-time anomaly insights

5. **Deployment**
   - Containerized with Docker for reproducibility

---

## 🏗️ System Architecture
Sensor Data → Preprocessing → Feature Scaling → ML Model → Predictions → Visualization


- **Data Layer:** CSV-based sensor dataset  
- **Processing Layer:** Pandas-based transformations  
- **Model Layer:** Scikit-learn (Isolation Forest)  
- **Serving Layer:** Python backend  
- **Presentation Layer:** Streamlit UI  

---

## 🤖 Model Details

### Algorithm: Isolation Forest

Isolation Forest isolates anomalies instead of profiling normal data.

**Why this approach:**
- No labeled dataset required
- Efficient for large datasets
- Robust to high-dimensional data

**Configuration:**
- `n_estimators = 100`
- `contamination = 0.1`
- `random_state = 42`

---

## 📂 Project Structure
.
├── backend/
│ ├── notebook.ipynb # Model training and experimentation
│ ├── model.pkl # Serialized model
│ ├── scaler.pkl # Feature scaler
│
├── data/
│ └── healthcare-wearable-vital-streams.csv
│
├── frontend/
│ └── app.py # Streamlit application
│
├── Dockerfile
├── requirements.txt
└── README.md


---

## ⚙️ Setup

### Clone Repository
```bash
git clone https://github.com/KhushalKhare/Sensor-data-analysis-
cd Sensor-data-analysis-

