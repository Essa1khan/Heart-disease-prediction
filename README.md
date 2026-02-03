# ❤️ Heart Disease Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Supervised-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-90.76%25-brightgreen.svg)

An end-to-end machine learning system that predicts heart disease with **90.76% accuracy** using K-Nearest Neighbors algorithm.

## 🎯 Project Overview

This project analyzes patient health data to predict the likelihood of heart disease. Built as part of supervised learning training (Jan 29 - Feb 3, 2026).

**Key Achievement:** 94.12% Recall - catches 94% of all heart disease cases!

## 📊 Dataset

- **Source:** Kaggle Heart Failure Prediction Dataset
- **Size:** 918 patient records
- **Features:** 12 clinical measurements
- **Classes:** Binary (Heart Disease / No Heart Disease)

## 🚀 Features

- ✅ Interactive web application (Streamlit)
- ✅ Real-time disease prediction
- ✅ Risk level visualization
- ✅ Medical recommendations
- ✅ 90.76% prediction accuracy
- ✅ Production-ready model

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.8+
- pandas, numpy - Data processing
- scikit-learn - Machine learning
- matplotlib, seaborn - Visualization
- Streamlit - Web application

**Machine Learning:**
- K-Nearest Neighbors (KNN) - Best model
- Also tested: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree, Gradient Boosting

## 📈 Model Performance

| Model | Accuracy | Recall | Status |
|-------|----------|--------|--------|
| **K-Nearest Neighbors** | **90.76%** | **94.12%** | ✅ Selected |
| Naive Bayes | 89.67% | 91.18% | Good |
| Logistic Regression | 89.13% | 92.16% | Good |
| Random Forest | 88.59% | 89.22% | Overfitting |

## 🎓 Key Learnings

**Feature Engineering:**
- Created `Cholesterol_Missing` indicator feature
- Converted missing data into valuable signal
- Result: 3rd strongest predictor (+0.32 correlation)

**Critical Insights:**
- Asymptomatic patients = Highest risk (79% disease rate)
- Male patients 2.4x more likely to have disease
- Oldpeak & MaxHR strongest predictors

**Best Practices:**
- Train-test split with stratification
- Feature scaling (StandardScaler)
- Overfitting detection and prevention
- Medical context in metric selection (Recall > Precision)

## 🖥️ Installation & Usage

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Access
```
Local URL: http://localhost:8501
```

## 📁 Project Structure
```
heart-disease-prediction/
│
├── app.py                 # Streamlit web application
├── model.pkl              # Trained KNN model
├── scaler.pkl             # Feature scaler
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🔍 How It Works

1. **User Input:** Patient enters 11 health parameters
2. **Preprocessing:** Data encoded and scaled
3. **Prediction:** KNN model predicts disease probability
4. **Output:** Risk level + recommendations displayed

## 🎯 Model Training Process
```python
# Data pipeline
Raw Data → Missing Value Handling → Feature Engineering 
→ Encoding → Scaling → Train-Test Split 
→ Model Training (7 algorithms) → Best Model Selection (KNN)
→ Deployment
```

## 📊 Sample Predictions

**High Risk Patient:**
```
Age: 65, Sex: Male, ChestPain: Asymptomatic
→ Prediction: 92% Disease Risk ⚠️
```

**Low Risk Patient:**
```
Age: 35, Sex: Female, ChestPain: Atypical
→ Prediction: 15% Disease Risk ✅
```

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Essa**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Project Duration: Jan 29 - Feb 3, 2026

## 🙏 Acknowledgments

- Mentor: Abdullah
- Dataset: Kaggle Community
- Framework: Streamlit Team

## ⚠️ Disclaimer

This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.

---

**Built with ❤️ using Machine Learning**
