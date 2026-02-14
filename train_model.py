import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Load and Clean Data
try:
    df = pd.read_csv(r'D:/MSC/Unemployment rate analysis/Survay_Responses.csv', encoding='latin1')
    # Remove any completely empty rows
    df = df.dropna(how='all')
except FileNotFoundError:
    print("Error: Could not find the CSV file. Check your file path.")
    exit()

# 2. Preprocessing Mappings
age_map = {'18–24': 1, '25–34': 2}
edu_map = {'School level': 1, 'Undergraduate': 2, 'Postgraduate': 3}
emp_map = {'Student': 1, 'Employed': 2, 'Unemployed': 3}

# Safely map values and fill missing with defaults
df['age_numeric'] = df['age_group'].map(age_map).fillna(1)
df['edu_numeric'] = df['education_level'].map(edu_map).fillna(2)
df['emp_numeric'] = df['employment_status'].map(emp_map).fillna(1)

# 3. Target Scoring
severity_map = {'Yes': 5, 'Maybe': 3, 'No': 1}
df['target_score'] = df['issue_severity_perception'].map(severity_map).fillna(3)

# 4. Encoding categorical variables
categorical_cols = ['gender', 'skill_alignment', 'skill_training', 'job_seeking_status']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Convert to string and strip whitespace to prevent encoding errors
    df[col] = le.fit_transform(df[col].astype(str).str.strip())
    encoders[col] = le

# 5. Define Features and Train Model
features = ['age_numeric', 'edu_numeric', 'emp_numeric', 'gender', 'skill_alignment', 'skill_training', 'job_seeking_status']
X = df[features]
y = df['target_score']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Save model and metadata
with open('unemployment_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model, 
        'encoders': encoders, 
        'age_map': age_map,
        'edu_map': edu_map,
        'emp_map': emp_map,
        'feature_names': features
    }, f)

print("Success: unemployment_model.pkl has been created!")