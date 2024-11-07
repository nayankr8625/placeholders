import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Preprocess the data by setting the identifier as the index and dropping unnecessary columns
def preprocess_data(data, index_col='Unnamed: 0', identifier_col='CADR ID'):
    data = data.set_index(index_col)
    data = data.drop(columns=[identifier_col], errors='ignore')
    return data

# Build the model pipeline
def build_model_pipeline(categorical_cols, numerical_cols):
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
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return model_pipeline

# Train the model
def train_model(model_pipeline, data):
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, X_train.index)
    return model_pipeline, X_train, X_test

# Calculate similarity using the proximity matrix
def calculate_similarity(model_pipeline, X_train, entries, top_n=10):
    entries_transformed = model_pipeline.named_steps['preprocessor'].transform(entries)
    train_leaf_indices = model_pipeline.named_steps['rf'].apply(model_pipeline.named_steps['preprocessor'].transform(X_train))
    entries_leaf_indices = model_pipeline.named_steps['rf'].apply(entries_transformed)

    proximity = np.zeros((len(entries_leaf_indices), len(train_leaf_indices)))
    for i, entry_leaf in enumerate(entries_leaf_indices):
        for j, train_leaf in enumerate(train_leaf_indices):
            proximity[i, j] = np.mean(entry_leaf == train_leaf)

    similar_entries_results = {}
    for idx, proximity_row in enumerate(proximity):
        similar_indices = np.argsort(-proximity_row)[:top_n]
        similar_entries_results[entries.index[idx]] = {
            X_train.index[i]: round(proximity_row[i] * 100, 2) for i in similar_indices
        }

    return similar_entries_results

# Main code for loading data, preprocessing, model training, and testing
if __name__ == "__main__":
    # Load and preprocess the data
    data_path = 'path_to_file.csv'
    data = pd.read_csv(data_path)
    data = preprocess_data(data)

    # Identify columns
    categorical_cols = [col for col in data.select_dtypes(include=['object']).columns]
    numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns]

    # Build and train the model
    model_pipeline = build_model_pipeline(categorical_cols, numerical_cols)
    model_pipeline, X_train, _ = train_model(model_pipeline, data)

    # Example usage: Single entry direct from dataset
    single_entry = X_train.iloc[[0]]
    similarities = calculate_similarity(model_pipeline, X_train, single_entry)
    print("Similarities for single direct entry:", similarities)

    # Example usage: Mixed entries
    mixed_entries = X_train.sample(n=5, random_state=42)  # Directly from training set
    new_entries = pd.DataFrame([X_train.iloc[0] * (1 + np.random.rand(len(X_train.columns)) * 0.1)], index=['New Entry'])  # Simulated new entry
    all_entries = pd.concat([mixed_entries, new_entries])
    mixed_similarities = calculate_similarity(model_pipeline, X_train, all_entries)
    print("Mixed entry similarities:", mixed_similarities)
