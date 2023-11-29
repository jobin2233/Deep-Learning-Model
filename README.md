                                                                                   # Deep-Learning-Model
# Lung Cancer Prediction Model

## Overview
This project aims to develop an AI model for assessing the risk of lung cancer in patients. The model predicts the likelihood of a patient being at low, medium, or high risk based on various features. The dataset includes information such as age, gender, air pollution exposure, lifestyle factors, and medical history.

## Dataset
The dataset (`cancer patient data sets.csv`) is loaded into a Pandas DataFrame (`df`). The features include:

# Features

## Personal Information.
- Age
- Gender

## Environmental Factors
- Air Pollution
- Alcohol Use
- Dust Allergy
- Occupational Hazards

## Genetic and Medical History
- Genetic Risk
- Chronic Lung Disease

## Lifestyle Factors
- Balanced Diet
- Obesity
- Smoking
- Passive Smoker

## Symptoms
- Chest Pain
- Coughing of Blood
- Fatigue
- Weight Loss
- Shortness of Breath
- Wheezing
- Swallowing Difficulty
- Clubbing of Finger Nails
- Frequent Cold
- Dry Cough
- Snoring

The target variable is 'Level,' indicating the risk category.

## Data Preprocessing
- The data is preprocessed using `StandardScaler` for numerical features.
- The target variable is encoded using `LabelEncoder` for training and testing sets.
- The data is split into training and testing sets.

## Model Architecture
The neural network model is implemented using TensorFlow and Keras. The model architecture consists of three layers:

1. **Input Layer:** BatchNormalization for input features.
2. **Hidden Layers:** Two dense layers with ReLU activation, BatchNormalization, and Dropout for regularization.
3. **Output Layer:** Dense layer with softmax activation for multiclass classification (low, medium, high risk).

The model is compiled with the Adam optimizer and categorical crossentropy loss function, suitable for multiclass classification.

## Training
The model is trained on the training data with early stopping to prevent overfitting. Training progress is monitored using accuracy and loss metrics.

## Evaluation
The model's performance is evaluated on the validation set, and the training history is visualized using Matplotlib. The loss and accuracy curves provide insights into the model's learning process.

