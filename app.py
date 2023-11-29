#import statements
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'D:\LungCancerPrediction\cancerpatientdatasets.csv')

#Headings
st.title('Lung Cancer Prediction')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

#X and y data
X = df.copy().dropna()
X.drop('index', axis=1, inplace=True)
X.drop('Patient Id', axis=1, inplace=True)
y = X.pop('Level')

features_num = ['Age', 'Gender', 'Air Pollution', 'Alcohol use',
       'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
       'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
       'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',
       'Weight Loss', 'Shortness of Breath', 'Wheezing',
       'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
       'Dry Cough', 'Snoring']

#Data pre-processing
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
)

#spliting the data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9598)

# Encode the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

input_shape = [X_train.shape[1]]

from keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Use categorical crossentropy for multiclass classification
    metrics=['accuracy'],  # Use 'accuracy' for multiclass classification
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train_one_hot,
    validation_data=(X_test, y_test_one_hot),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

def user_report():
    age = st.sidebar.slider('Age', 10, 88, 30)
    gender = st.sidebar.slider('Gender', 1, 2, 2)
    airpollution = st.sidebar.slider('Air Pollution', 1, 10, 3)
    alcoholuse = st.sidebar.slider('Alcohol use', 1, 10, 3)
    dustallergy = st.sidebar.slider('Dust Allergy', 1, 10, 3)
    occupationalhazards = st.sidebar.slider('OccuPational Hazards', 1, 10, 3)
    geneticrisk = st.sidebar.slider('Genetic Risk', 1, 10, 3)
    chroniclungdisease = st.sidebar.slider('chronic Lung Disease', 1, 10, 9)
    balanceddiet = st.sidebar.slider('Balanced Diet', 1, 10, 3)
    obesity = st.sidebar.slider('Obesity', 1, 10, 3)
    smoking = st.sidebar.slider('Smoking', 1, 10, 3)
    passivesmoker = st.sidebar.slider('Passive Smoker', 1, 10, 3)
    chestpain = st.sidebar.slider('Chest Pain', 1, 10, 3)
    coughingofblood = st.sidebar.slider('Coughing of Blood', 1, 10, 3)
    fatigue = st.sidebar.slider('Fatigue', 1, 10, 3)
    weightloss = st.sidebar.slider('Weight Loss', 1, 10, 3)
    shortnessofbreath = st.sidebar.slider('Shortness of Breath', 1, 10, 3)
    wheezing = st.sidebar.slider('Wheezing', 1, 10, 3)
    swallowingdifficulty = st.sidebar.slider('Swallowing Difficulty', 1, 10, 3)
    clubbingoffingernails = st.sidebar.slider('Clubbing of Finger Nails', 1, 10, 3)
    frequentcold = st.sidebar.slider('Frequent Cold', 1, 10, 3)
    drycough = st.sidebar.slider('Dry Cough', 1, 10, 6)
    snoring = st.sidebar.slider('Snoring', 1, 10, 3)
    user_data = {
        'age': age,
        'gender': gender,
        'airpollution': airpollution,
        'alcoholuse': alcoholuse,
        'dustallergy': dustallergy,
        'occupationalhazards': occupationalhazards,
        'geneticrisk': geneticrisk,
        'chroniclungdisease': chroniclungdisease,
        'balanceddiet': balanceddiet,
        'obesity': obesity,
        'smoking': smoking,
        'passivesmoker': passivesmoker,
        'chestpain': chestpain,
        'coughingofblood': coughingofblood,
        'fatigue': fatigue,
        'weightloss': weightloss,
        'shortnessofbreath': shortnessofbreath,
        'wheezing': wheezing,
        'swallowingdifficulty': swallowingdifficulty,
        'clubbingoffingernails': clubbingoffingernails,
        'frequentcold': frequentcold,
        'drycough': drycough,
        'snoring': snoring
    }
    
    user_df = pd.DataFrame([user_data], columns=features_num)

    user_input_transformed = preprocessor.transform(user_df)

    return user_input_transformed

user_data = user_report()

user_result_probabilities = model.predict(user_data)
predicted_class = np.argmax(user_result_probabilities, axis=1)

st.subheader('Your Report: ')
output = ''
if predicted_class[0] == 0:
    output = 'low'
elif predicted_class[0] == 1:
    output = 'medium'
else:
    output = 'high'

st.write(output)
