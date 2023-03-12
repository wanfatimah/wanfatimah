import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Advertising Sales Prediction App

This app predicts the **Advertising** Sales!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 5.4)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 1.3)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

advertising = datasets.load_advertising()
X = advertising.data
Y = advertising.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(advertising.target_names)

st.subheader('Prediction')
st.write(advertising.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
