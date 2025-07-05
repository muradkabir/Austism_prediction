# Autism Spectrum Disorder Prediction App

A Streamlit web application for predicting the likelihood of Autism Spectrum Disorder (ASD) based on behavioral features and demographic information.

## Overview

This application uses a machine learning model to predict the likelihood of ASD based on the following features:
- 10 behavioral questions (A1_Score through A10_Score)
- Demographic information (age, gender, ethnicity, etc.)
- Family history (jaundice, family members with autism)

The model has been trained on a dataset containing information about individuals with and without ASD diagnosis.

## Features

- **Prediction Tool**: Get an instant prediction on the likelihood of ASD based on user inputs
- **Data Insights**: Explore dataset statistics, distributions, and feature importance
- **Interactive Visualizations**: Graphs and charts to understand the dataset and prediction results
- **User-friendly Interface**: Simple and intuitive UI designed for ease of use

## Installation

1. Clone this repository:
```
git clone [repository-url]
cd autism-prediction
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Dataset

The app uses the Autism Screening Adult dataset with the following variables:
- A1_Score to A10_Score: Answers to the 10 questions on the AQ-10 screening test (0 = No, 1 = Yes)
- age: Age of the individual in years
- gender: Gender (m = Male, f = Female)
- ethnicity: Ethnicity of the individual
- jaundice: Whether the individual was born with jaundice (yes/no)
- austim: Whether the individual has a family member with autism (yes/no)
- contry_of_res: Country of residence
- used_app_before: Whether the individual has used a screening app before (yes/no)
- result: Total score on the AQ-10 test
- age_desc: Age description (e.g., "18 and more")
- relation: Relation of the person completing the questionnaire
- Class/ASD: Target variable (1 = ASD, 0 = No ASD)

## Model

The application uses a logistic regression model trained on the dataset. The model takes in the behavioral and demographic features and outputs a probability score representing the likelihood of ASD.

## Notes

- This application is for educational and research purposes only
- It is not a substitute for professional medical advice, diagnosis, or treatment
- Please consult with a healthcare professional for proper evaluation
