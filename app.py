import streamlit as st

import joblib
import pandas as pd

model = joblib.load("placement_model.pkl")

st.title("AI Powered Placement Prediction System")
st.divider()

cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)

internships = st.number_input("Internships", 0, 20, 1)

projects = st.number_input("Projects", 0, 20, 2)

workshops = st.number_input(
    "Workshops/Certifications",
    0,
    20,
    2
)

aptitude = st.number_input(
    "Aptitude Test Score",
    0,
    100,
    70
)

softskills = st.number_input(
    "Soft Skills Rating",
    0.0,
    10.0,
    7.0
)

extra = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

training = st.selectbox(
    "Placement Training",
    ["Yes", "No"]
)

ssc = st.number_input(
    "SSC Marks",
    0,
    100,
    80
)

hsc = st.number_input(
    "HSC Marks",
    0,
    100,
    80
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "CGPA":[cgpa],
        "Internships":[internships],
        "Projects":[projects],
        "Workshops/Certifications":[workshops],
        "AptitudeTestScore":[aptitude],
        "SoftSkillsRating":[softskills],
        "ExtracurricularActivities":[extra],
        "PlacementTraining":[training],
        "SSC_Marks":[ssc],
        "HSC_Marks":[hsc]
    })

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)

    placed_index = list(model.classes_).index("Placed")

    placement_probability = probabilities[0][placed_index] * 100

    st.divider()
    if prediction == "Placed":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    st.metric(
    "Placement Probability",
    f"{placement_probability:.2f}%"
    )