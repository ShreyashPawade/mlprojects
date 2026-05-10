import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    # Page configuration
    st.set_page_config(page_title="Student Performance Predictor")
    
    st.title("Student Exam Performance Prediction")
    st.markdown("Enter the student details below to predict their math score.")

    # --- Sidebar or Main Panel Inputs ---
    st.header("Student Information")
    
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        race_ethnicity = st.selectbox(
            "Race/Ethnicity", 
            ["group A", "group B", "group C", "group D", "group E"]
        )
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            [
                "associate's degree", 
                "bachelor's degree", 
                "high school", 
                "master's degree", 
                "some college", 
                "some high school"
            ]
        )

    with col2:
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_preparation_course = st.selectbox(
            "Test Preparation Course", 
            ["none", "completed"]
        )
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=70)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=70)

    # --- Prediction Logic ---
    if st.button("Predict your Maths Score"):
        try:
            # Initialize Data Class
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(reading_score),
                writing_score=float(writing_score)
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            
            st.subheader("Input Dataframe")
            st.write(pred_df)

            # Prediction Pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Display Results
            st.success(f"The predicted Maths Score is: **{results[0]:.2f}**")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
