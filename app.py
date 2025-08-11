import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Autism Spectrum Disorder Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_autism_model():
    return load_model('autism_detection_model.h5')

model = load_autism_model()

# Print model input shape for debugging
print("Model input shape:", model.input_shape)

# Load dataset for statistics
@st.cache_data
def load_data():
    return pd.read_csv('train.csv')

df = load_data()

# Define functions
def predict_autism(features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    probability = prediction[0][0]
    return probability, 1 if probability > 0.5 else 0

# Main app interface
st.title("Autism Spectrum Disorder Prediction")
st.write("This app predicts the likelihood of Autism Spectrum Disorder based on behavioral and demographic features.")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Prediction", "Data Insights"])

if page == "Prediction":
    st.header("Prediction Tool")
    st.write("Please fill in the following information to get a prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Behavioral Features")
        a1 = st.selectbox("A1: I often notice small sounds when others do not", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a2 = st.selectbox("A2: I usually concentrate more on the whole picture, rather than the small details", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a3 = st.selectbox("A3: I find it easy to do more than one thing at once", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a4 = st.selectbox("A4: If there is an interruption, I can switch back to what I was doing very quickly", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a5 = st.selectbox("A5: I find it easy to 'read between the lines' when someone is talking to me", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a6 = st.selectbox("A6: I know how to tell if someone listening to me is getting bored", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a7 = st.selectbox("A7: When I'm reading a story I find it difficult to work out the characters' intentions", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a8 = st.selectbox("A8: I like to collect information about categories of things", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a9 = st.selectbox("A9: I find it easy to work out what someone is thinking or feeling just by looking at their face", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        a10 = st.selectbox("A10: I find it difficult to work out people's intentions", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    with col2:
        st.subheader("Demographic Information")
        age = st.number_input("Age", min_value=1.0, max_value=100.0, step=0.1)
        gender = st.selectbox("Gender", ["m", "f"], format_func=lambda x: "Male" if x == "m" else "Female")
        gender_encoded = 1 if gender == "m" else 0
        
        jaundice = st.selectbox("Born with jaundice?", ["yes", "no"], format_func=lambda x: "Yes" if x == "yes" else "No")
        jaundice_encoded = 1 if jaundice == "yes" else 0
        
        autism_family = st.selectbox("Family member with autism?", ["yes", "no"], format_func=lambda x: "Yes" if x == "yes" else "No") 
        autism_family_encoded = 1 if autism_family == "yes" else 0
        
        ethnicity_options = ["White-European", "Asian", "Middle Eastern", "Black", "Hispanic", "Latino", "South Asian", "Turkish", "Others"]
        ethnicity = st.selectbox("Ethnicity", ethnicity_options)
        
        # Create dummy variables for ethnicity (simplified)
        ethnicity_encoded = ethnicity_options.index(ethnicity) / len(ethnicity_options)
        
        # Add country of residence field
        country_options = ["United States", "United Kingdom", "India", "Australia", "Canada", "Others"]
        country = st.selectbox("Country of Residence", country_options)
        country_encoded = country_options.index(country) / len(country_options)
        
        used_app_before = st.selectbox("Used a screening app before?", ["yes", "no"], format_func=lambda x: "Yes" if x == "yes" else "No")
        used_app_encoded = 1 if used_app_before == "yes" else 0
        
        relation_options = ["Self", "Parent", "Relative", "Others"]
        relation = st.selectbox("Who is completing this form?", relation_options)
        relation_encoded = relation_options.index(relation) / len(relation_options)
    
    # Calculate result score (simplified)
    result_score = sum([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])
    
    # Predict button
    if st.button("Predict"):
        # Prepare features
        features = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, 
                   age, gender_encoded, ethnicity_encoded, jaundice_encoded, 
                   autism_family_encoded, country_encoded, used_app_encoded, result_score, relation_encoded]
        
        probability, prediction = predict_autism(features)
        
        # Display prediction
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ASD Probability", f"{probability:.2%}")
            st.write(f"**Prediction:** {'ASD Positive' if prediction == 1 else 'ASD Negative'}")
            
            if prediction == 1:
                st.error("The model predicts a higher likelihood of Autism Spectrum Disorder.")
                
            else:
                st.success("The model predicts a low likelihood of Autism Spectrum Disorder.")
                
            # Display model accuracy information
            st.info("Model accuracy: 88% (Based on validation data)")
                
        with col2:
            # Create gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ASD Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "coral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)

elif page == "Data Insights":
    st.header("Dataset Insights")
    
    # General statistics
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", len(df))
        asd_positive = len(df[df["Class/ASD"] == 1])
        st.metric("ASD Positive Cases", asd_positive)
        
    with col2:
        st.metric("Features", len(df.columns))
        st.metric("ASD Prevalence", f"{asd_positive/len(df):.2%}")
    
    # Distribution of ASD cases
    st.subheader("Distribution of ASD Cases")
    fig = px.pie(df, names="Class/ASD", color="Class/ASD",
                color_discrete_map={0: 'lightblue', 1: 'darkblue'},
                labels={0: 'Non-ASD', 1: 'ASD'})
    st.plotly_chart(fig)
    
    # Age Distribution
    st.subheader("Age Distribution")
    fig = px.histogram(df, x="age", color="Class/ASD", 
                      labels={"Class/ASD": "Autism Status", "age": "Age"},
                      color_discrete_map={0: 'lightblue', 1: 'darkblue'},
                      barmode="overlay")
    st.plotly_chart(fig)
    
    # Gender Distribution
    st.subheader("Gender Distribution")
    gender_counts = df.groupby(['gender', 'Class/ASD']).size()
    gender_counts = gender_counts.reset_index()
    gender_counts.columns = ['gender', 'Class/ASD', 'count']
    fig = px.bar(gender_counts, x="gender", y="count", color="Class/ASD",
                labels={"Class/ASD": "Autism Status", "gender": "Gender", "count": "Count"},
                color_discrete_map={0: 'lightblue', 1: 'darkblue'})
    st.plotly_chart(fig)
    
    # Feature importance (simplified)
    st.subheader("Behavioral Feature Response Frequencies")
    behavioral_features = [f'A{i}_Score' for i in range(1, 11)]
    
    feature_data = []
    for feature in behavioral_features:
        positive_rate = df[df[feature] == 1]['Class/ASD'].mean()
        feature_data.append({'Feature': feature, 'Positive Response Rate in ASD': positive_rate})
    
    feature_df = pd.DataFrame(feature_data)
    fig = px.bar(feature_df, x='Feature', y='Positive Response Rate in ASD',
                labels={"Feature": "Question", "Positive Response Rate in ASD": "ASD Positive Rate"},
                color='Positive Response Rate in ASD', color_continuous_scale='Blues')
    st.plotly_chart(fig)
    
    # Ethnicity Analysis
    if st.checkbox("Show Ethnicity Analysis"):
        st.subheader("ASD by Ethnicity")
        ethnicity_counts = df.groupby(['ethnicity', 'Class/ASD']).size()
        ethnicity_counts = ethnicity_counts.reset_index()
        ethnicity_counts.columns = ['ethnicity', 'Class/ASD', 'count']
        ethnicity_counts = ethnicity_counts[ethnicity_counts['ethnicity'] != '?']
        fig = px.bar(ethnicity_counts, x="ethnicity", y="count", color="Class/ASD",
                    labels={"Class/ASD": "Autism Status", "ethnicity": "Ethnicity", "count": "Count"},
                    color_discrete_map={0: 'lightblue', 1: 'darkblue'})
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.write("Jahangirnagar University")


