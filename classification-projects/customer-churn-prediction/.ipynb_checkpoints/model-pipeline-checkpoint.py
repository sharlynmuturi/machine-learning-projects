import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

import joblib

df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop("customerID", axis=1, inplace=True)

df = df[df.TotalCharges!=' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Normalizing text categories
df = df.replace('No internet service', 'No')
df = df.replace('No phone service', 'No')

X = df.drop("Churn", axis=1)
y = df["Churn"].replace({'Yes': 1, 'No': 0})

# Building Transformers

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
binary_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
gender_feature = ['gender']
categorical_features = ['InternetService', 'Contract', 'PaymentMethod']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('bin', OneHotEncoder(drop='if_binary'), binary_features + gender_feature),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

smote = SMOTE(random_state=42)

# Training a Random Forest model

model = RandomForestClassifier(random_state=42)

# Training the pipeline
pipeline = ImbPipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),      # Only applied during fit
        ('classifier', model)
    ]
)

pipeline.fit(X, y)

joblib.dump(pipeline, "churn_pipeline.pkl")