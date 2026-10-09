# 🏥 Medical Appointment No-Show Prediction System

An end-to-end Machine Learning web application designed to predict patient appointment attendance and mitigate operational healthcare losses. The system utilizes an optimized LightGBM classifier with class weight balancing and Recursive Feature Elimination (RFE) for real-time inference.

---

## 📸 Overview & Key Features

- **Attendance Risk Classification**: Predicts whether a patient will attend (`Show`) or miss (`No-Show`) their scheduled medical appointment.
- **Recursive Feature Elimination (RFE)**: Filters down high-dimensional patient demographic and scheduling records into the 13 most impactful features.
- **Class Imbalance Resolution**: Handles target class skewness using dynamic class weight balancing to preserve recall for no-show events.
- **Calculated Clinical Metrics**: Engineers custom features such as `WaitingDays`, `AppointmentWeekday`, `ScheduledWeekday`, and chronic `DiseaseCount`.
- **Interactive Streamlit Web Dashboard**: Real-time evaluation portal providing instant attendance probabilities, confidence metrics, and actionable clinical recommendations.

---

## 🛠️ Tech Stack & Dependencies

- **Frontend / Application Framework**: Streamlit
- **Machine Learning & Feature Selection**: LightGBM, Scikit-Learn (RFE), XGBoost, CatBoost, Random Forest
- **Data Manipulation & Preprocessing**: Pandas, NumPy
- **Class Imbalance Management**: Imbalanced-Learn, Sample/Class Weight Balancing
- **Visualization & EDA**: Matplotlib, Seaborn

---

## 📁 Repository Structure

No_Show_Prediction/
├── app.py                            # Interactive Streamlit web application
├── no_show_prediction.ipynb          # ML Pipeline (EDA, Feature Engineering, Tuning, RFE, Evaluation)
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git exclusion rules

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
git clone https://github.com/habiba-10/Medical-Appointment-No-Show-Prediction.git
cd Medical-Appointment-No-Show-Prediction

### 2. Set Up Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run the Streamlit Application
streamlit run app.py

---

## ⚙️ Model Architecture & Pipeline Workflow

1. **Exploratory Data Analysis & Cleaning**: Cleaned invalid age entries and handled temporal scheduling formats.
2. **Feature Engineering**: Extracted lead-time duration (`WaitingDays`), day-of-week indicators, and aggregated chronic conditions (`DiseaseCount`).
3. **Hyperparameter Tuning & Selection**: Evaluated Random Forest, XGBoost, CatBoost, and LightGBM across cross-validation folds.
4. **Recursive Feature Elimination (RFE)**: Selected top 13 predictive features to streamline runtime inference.
5. **Class Imbalance Optimization**: Applied class weighting to balance accuracy and recall for high-risk no-shows.

---

## 👤 Author
**Habiba Ezat**
- GitHub: [@habiba-10](https://github.com/habiba-10)


