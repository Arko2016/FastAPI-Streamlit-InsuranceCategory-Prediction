import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

#load data
input_data_path = './datasets/insurance.csv'
input_df = pd.read_csv(input_data_path)
#print(input_df.sample(5))

#create a copy of dataset which will be used for data preprocessing and feature engineering
df_feat = input_df.copy()

#check unique occupation levels
#print(df_feat['occupation'].unique())

#feature 1: calculate bmi using height and weight
df_feat['bmi'] = round(df_feat['weight']/(df_feat['height'] ** 2),2)

#feature 2: create age buckets
def age_group_func(age):
    if age < 25:
        return 'young'
    elif age < 40:
        return 'adult'
    elif age < 65:
        return 'middle-aged'
    else:
        return 'senior'

df_feat['age_group'] = df_feat['age'].apply(age_group_func)

#feature 3: create lifestyle risk attribute based on bmi and smoker flag
def lifestyle_risk_func(record):
    if record['smoker'] and record['bmi'] > 30:
        return 'high'
    elif record['smoker'] and record['bmi'] > 27:
        return 'medium'
    else:
        return 'low'

df_feat['lifestyle_risk'] = df_feat.apply(lifestyle_risk_func, axis = 1)

#feature 4: bucket cities to tiers
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

def city_tier_func(city):
    if city in tier_1_cities:
        return 'tier1'
    elif city in tier_2_cities:
        return 'tier2'
    else:
        return 'tier3'
df_feat['city_tier'] = df_feat['city'].apply(city_tier_func)

#view the dataframe with only essential columns
#print(df_feat[['income_lpa', 'occupation', 'bmi', 'age_group', 'lifestyle_risk', 'city_tier', 'insurance_premium_category']].sample(5))

#split to predictors and target attributes
X = df_feat[['income_lpa', 'occupation', 'bmi', 'age_group', 'lifestyle_risk', 'city_tier']]
y = df_feat['insurance_premium_category']

#specify categorical and numeric predictors
cat_features = ["age_group", "lifestyle_risk", "occupation", "city_tier"]
num_features = list(set(X.columns.tolist()) - set(cat_features))

#create a column transformer object to one-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(),cat_features),
        ('num','passthrough',num_features)
    ]
)

#create a pipeline for preprocessing followed by RandomForest classifier
pipeline = Pipeline(steps = [
    ('preprocesser',preprocessor),
    ('classifer',RandomForestClassifier(random_state=42))
])

#split data to test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

#fit the model on train data using pipeline built above
pipeline.fit(X_train,y_train)
#print(pipeline)

#generate predictions using pipeline and evaluate model
y_pred = pipeline.predict(X_test)
#print(accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

#save trained pipeline using pickle
pickle_model_path = './pickle_files/model.pkl'
with open(pickle_model_path,"wb") as f: #need to write the  file in binary mode since pickle is a binary file
    pickle.dump(pipeline,f)