# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 12:56:00 2025

@author: renuka
"""
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# Prepare training data
train_df = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]

# Prepare test data — no Survived column
test_df = test_df[['Pclass', 'Sex', 'Age', 'Fare']].dropna()

# Preprocessing pipeline
numeric_features = ['Age', 'Fare']
numeric_transformer = StandardScaler()

categorical_features = ['Pclass', 'Sex']
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Streamlit App
st.title("🚢 Titanic Survival Predictor")
st.header("🧮 Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 100, 25)
fare = st.number_input("Fare", min_value=0.0, step=0.1, value=32.2)

user_input = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Fare': [fare]
})

if st.button("Predict Survival"):
    prediction = pipeline.predict(user_input)[0]
    prob = pipeline.predict_proba(user_input)[0][1]

    st.subheader("🎯 Prediction Result:")
    st.success("Survived ✅" if prediction == 1 else "Did Not Survive ❌")
    st.info(f"Survival Probability: **{prob:.2f}**")

