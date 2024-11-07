import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cosine
import random

def preprocess_data(data, identifier_col='CADR ID'):
    """
    Preprocess data by setting a material identifier as the index and dropping unnecessary columns.
    """
    data = data.set_index('Unnamed: 0')
    data = data.drop(columns=[identifier_col], errors='ignore')
    return data

def build_model_pipeline(categorical_cols, numerical_cols, n_components=20):
    """
    Builds a model pipeline with preprocessing, dimensionality reduction, and a random forest model.
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svd', TruncatedSVD(n_components=n_components, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return model_pipeline

def train_model(model_pipeline, data):
    """
    Splits the data and trains the model pipeline.
    """
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, X_train.index)
    return model_pipeline, X_train, X_test

def find_similar_entries(model_pipeline, X_train, entry_data, top_n=10):
    """
    Finds similar entries to a given entry based on cosine similarity with the SVD-transformed data.
    """
    transformed_entry = model_pipeline.named_steps['preprocessor'].transform(pd.DataFrame([entry_data]))
    transformed_entry = model_pipeline.named_steps['svd'].transform(transformed_entry).flatten()
    
    similarities = {}
    for idx, train_entry in enumerate(model_pipeline.named_steps['svd'].transform(model_pipeline.named_steps['preprocessor'].transform(X_train))):
        similarity_score = 1 - cosine(transformed_entry, train_entry.flatten())
        similarities[X_train.index[idx]] = round(similarity_score * 100, 2)
    
    similar_entries = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n])
    return similar_entries

def direct_entry_accuracy_test(model_pipeline, X_train, top_n=10, num_samples=20):
    """
    Tests the accuracy of the model by checking if real entries are in the top_n results.
    """
    matches = 0
    sample_indices = X_train.sample(num_samples, random_state=42).index
    
    for sample_index in sample_indices:
        entry_data = X_train.loc[sample_index].to_dict()
        similar_entries = find_similar_entries(model_pipeline, X_train, entry_data, top_n=top_n)
        
        if sample_index in similar_entries:
            matches += 1

    accuracy = matches / num_samples * 100
    return accuracy

# Usage
# Load and preprocess the data
file_path = 'path_to_file.csv'
data = pd.read_csv(file_path)
data = preprocess_data(data)

# Identify columns
categorical_cols = [col for col in data.select_dtypes(include=['object']).columns]
numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns]

# Build and train the model
model_pipeline = build_model_pipeline(categorical_cols, numerical_cols)
model_pipeline, X_train, X_test = train_model(model_pipeline, data)

# Test for accuracy using direct entries
accuracy = direct_entry_accuracy_test(model_pipeline, X_train, top_n=10, num_samples=20)
print(f"Direct entry accuracy: {accuracy}%")

# Example usage to find similar entries for a new or modified entry
sample_entry = X_train.iloc[0].to_dict()
similar_entries = find_similar_entries(model_pipeline, X_train, sample_entry, top_n=10)
print("Similar entries:", similar_entries)
