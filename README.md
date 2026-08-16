# 🎓 Student Performance AI

An AI-based Student Performance Prediction and Academic Decision Support System using Machine Learning and Explainable AI.

## 📌 Overview

Student Performance AI is a machine-learning-based application designed to analyze student academic and personal indicators and predict their expected academic performance.

The system also provides academic decision-support features such as:

- Student performance prediction
- Performance classification
- Risk-level identification
- Student comparison
- What-if performance simulation
- AI intervention analysis
- Personalized recommendations
- 30-day student action plans
- High-risk student identification
- Intervention comparison
- Downloadable student action plans

## 🤖 Machine Learning

The project uses a Random Forest machine learning model for student performance prediction.

The model analyzes academic indicators and generates a predicted performance score.

### Main Concepts

- Data preprocessing
- Exploratory Data Analysis
- Feature analysis
- Machine Learning
- Random Forest
- Model evaluation
- Prediction
- Risk classification
- Explainable decision support

## 🧠 Key Features

### 1. Student Performance Prediction

Predicts the expected student performance based on the provided academic indicators.

### 2. Risk Classification

Students are classified into different risk categories to help identify students who may require additional academic support.

### 3. Student Comparison

Allows comparison between two students using:

- Predicted grade
- Performance level
- Risk level
- Study time
- Previous failures
- Absences
- Health
- Age

### 4. What-If Performance Simulator

Allows academic indicators to be changed and evaluates how the existing machine-learning model responds.

Example indicators include:

- Study time
- Absences
- Previous failures

### 5. AI Intervention Analysis

Tests simulated academic interventions such as:

- Increasing study time
- Improving attendance
- Addressing previous failures
- Combined interventions

### 6. Personalized Recommendations

Generates recommendations based on the student's current academic indicators.

### 7. 30-Day Student Action Plan

Creates a practical academic roadmap containing:

- Priority assessment
- Recommended focus areas
- Weekly academic goals
- Recommended actions
- Success checks

### 8. High-Risk Student Identification

Identifies students classified as high risk so that they can be prioritized for academic monitoring and support.

## 📊 Project Structure

```text
student-performance-analysis/
│
├── data/
│   ├── student-performance-original.csv
│   └── student-mat.csv
│
├── images/
│   ├── attendance_vs_final_score.png
│   ├── correlation_heatmap.png
│   ├── parental_education_vs_final_score.png
│   ├── random_forest_actual_vs_predicted.png
│   ├── random_forest_feature_importance.png
│   ├── study_hours_vs_final_score.png
│   ├── test_preparation_vs_final_score.png
│   └── top_factors_g3.png
│
├── models/
│   ├── model_comparison.csv
│   ├── model_features.pkl
│   ├── model_metrics.json
│   ├── model_metrics.pkl
│   └── student_performance_model.pkl
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── app.py
├── training_model.py
├── .gitignore
└── README.md