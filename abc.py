import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier



# Identify categorical and numerical columns but exclude the identifier if it's part of the DataFrame
categorical_cols = [col for col in data.select_dtypes(include=['object']).columns if col != 'YourIdentifierColumn']
numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns if col != 'YourIdentifierColumn']

# Creating preprocessing pipelines for both numerical and categorical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing steps into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# Setting up the SVD and KNN pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svd', TruncatedSVD(n_components=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# Splitting data into training and testing sets
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# Fitting the model
model_pipeline.fit(X_train, X_train.index)  # Use index for identifying records post-prediction

# Function to find similar entries
def find_similar_entries(entry_data, top_n=5):
    # Transform and predict the nearest neighbors for the entry data
    transformed_entry = model_pipeline.named_steps['preprocessor'].transform(pd.DataFrame([entry_data]))
    transformed_entry = model_pipeline.named_steps['svd'].transform(transformed_entry)
    distances, indices = model_pipeline.named_steps['knn'].kneighbors(transformed_entry, n_neighbors=top_n)
    
    # Calculate normalized similarity scores
    max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
    similarity_scores = {
        X_train.index[idx]: round((1 - dist / max_dist) * 100, 2) 
        for idx, dist in zip(indices[0], distances[0])
    }
    
    return similarity_scores

# Example usage
example_entry = X_train.iloc[0].to_dict()
similar_entries = find_similar_entries(example_entry)
for identifier, score in similar_entries.items():
    print(f"{identifier}: {score}% Similarity")
