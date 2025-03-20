import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Custom CSS to add background image
def add_background(image_file):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({image_file});
             background-size: cover;
             background-repeat: no-repeat;
             background-attachment: fixed;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Add background image of wheat and paddy fields
add_background('https://i.postimg.cc/kgHLg4YL/premium-photo-1698086768776-2fe137e167df.avif')

# Title of the app
st.title('Crop Recommendation: Wheat / Paddy')

st.info('This app uses a machine learning model to recommend the best crop (Wheat or Paddy) based on your input!')

# Dataset for demonstration purposes (expanded with new soil types)
data = {
    'soil_type': ['Loamy', 'Sandy', 'Clay', 'Loamy', 'Sandy', 'Clay', 'Black', 'Alluvial', 'Red'],
    'temperature': [20, 30, 25, 18, 32, 28, 25, 27, 22],
    'rainfall': [20, 15, 22, 18, 14, 16, 23, 21, 19],  # Rainfall in cm
    'humidity': [50, 65, 70, 45, 60, 55, 68, 52, 62],
    'nitrogen': [50, 40, 60, 55, 45, 65, 55, 60, 50],
    'phosphorus': [30, 35, 40, 20, 45, 25, 40, 35, 30],
    'sulphur': [20, 25, 30, 15, 35, 20, 28, 22, 25],
    'potassium': [40, 50, 55, 60, 45, 50, 60, 55, 52],
    'crop': ['Wheat', 'Paddy', 'Paddy', 'Wheat', 'Paddy', 'Wheat', 'Wheat', 'Paddy', 'Wheat']
}

df = pd.DataFrame(data)

with st.expander('Data'):
    st.write('Raw data used for training:')
    st.dataframe(df)

# Sidebar input for the user
with st.sidebar:
    st.header('Input Conditions for Your Farm')
    
    soil_type = st.selectbox('Soil Type', ('Loamy', 'Sandy', 'Clay', 'Black', 'Alluvial', 'Red'))
    temperature = st.slider('Temperature (°C)', 10, 45, 25)
    rainfall = st.slider('Rainfall (cm)', 5, 30, 15)  # Adjusted to cm
    humidity = st.slider('Humidity (%)', 30, 90, 60)
    
    nitrogen = st.slider('Nitrogen (N) level', 0, 100, 50)
    phosphorus = st.slider('Phosphorus (P) level', 0, 100, 30)
    sulphur = st.slider('Sulphur (S) level', 0, 100, 20)
    potassium = st.slider('Potassium (K) level', 0, 100, 40)

    # Create DataFrame for the input features
    input_data = {
        'soil_type': soil_type,
        'temperature': temperature,
        'rainfall': rainfall,
        'humidity': humidity,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'sulphur': sulphur,
        'potassium': potassium
    }
    input_df = pd.DataFrame(input_data, index=[0])

# Display user input
with st.expander('Input features'):
    st.write('Here are the conditions you provided:')
    st.dataframe(input_df)

# Data preparation
X_raw = df.drop('crop', axis=1)
y_raw = df['crop']

# One-hot encode categorical features (soil_type)
X_encoded = pd.get_dummies(X_raw, columns=['soil_type'])
input_encoded = pd.get_dummies(input_df, columns=['soil_type'])

# Ensure input_encoded has the same columns as X_encoded
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Encode the target variable (crop: Wheat=0, Paddy=1)
target_mapper = {'Wheat': 0, 'Paddy': 1}
y = y_raw.map(target_mapper)

# Model training
clf = RandomForestClassifier()
clf.fit(X_encoded, y)

# Check if input matches any row in the training data
matched = df[(df[['soil_type', 'temperature', 'rainfall', 'humidity', 'nitrogen', 'phosphorus', 'sulphur', 'potassium']] == list(input_data.values())).all(axis=1)]

# Prediction based on input data
if not matched.empty:
    predicted_crop = matched['crop'].values[0]
else:
    # Prediction if no match is found
    prediction = clf.predict(input_encoded)
    predicted_crop = 'Paddy' if (75 <= rainfall <= 100) else np.array(['Wheat', 'Paddy'])[prediction][0]

# Display prediction
st.subheader('Prediction Results')
if not matched.empty:
    st.success(f'Recommended Crop: {predicted_crop} (based on exact match with training data)')
else:
    st.success(f'Recommended Crop: {predicted_crop}')
