import streamlit as st
import pandas as pd

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

loaded_model = pickle.load(open("advertising.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
