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

# Prediction
def find_similar_entries(data, top_n=5):
    # Check and parse JSON input
    if isinstance(data, str):
        try:
            # Attempt to parse JSON string as DataFrame or as a single JSON record
            data = pd.read_json(data, orient='records')
        except ValueError:
            # Handle the case where the JSON data is a single object
            data = pd.DataFrame([json.loads(data)])
    
    # Ensure input is in DataFrame format
    if isinstance(data, pd.Series):
        data = pd.DataFrame([data])  # Convert a single series into a DataFrame with one row
    
    results = {}  # Dictionary to store results for each entry
    
    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Transform the current row
        transformed_row = model_pipeline.named_steps['preprocessor'].transform(pd.DataFrame([row]))
        transformed_row = model_pipeline.named_steps['svd'].transform(transformed_row)
        
        # Compute nearest neighbors
        distances, indices = model_pipeline.named_steps['knn'].kneighbors(transformed_row, n_neighbors=top_n)
        
        # Calculate normalized similarity scores
        max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
        similarity_scores = {
            X_train.index[idx]: round((1 - dist / max_dist) * 100, 2)
            for idx, dist in zip(indices[0], distances[0])
        }
        
        # Store similarity scores using the DataFrame index as key
        results[index] = similarity_scores
    
    return results

# Model Evaluation

from sklearn.cluster import KMeans


# Cluster the data to define 'true' similarities
kmeans = KMeans(n_clusters=10, random_state=42).fit(preprocessor.transform(data))
data['cluster_label'] = kmeans.labels_

# Group data by clusters to define true similarity groups
true_groups = data.groupby('cluster_label').apply(lambda x: list(x.index)).tolist()


def find_similar_entries(entry_data, top_n=10):
    transformed_entry = model_pipeline.named_steps['preprocessor'].transform(pd.DataFrame([entry_data]))
    transformed_entry = model_pipeline.named_steps['svd'].transform(transformed_entry)
    distances, indices = model_pipeline.named_steps['knn'].kneighbors(transformed_entry, n_neighbors=top_n)
    return [X_train.index[idx] for idx in indices[0]]

# Apply the function to test data
predictions = [find_similar_entries(X_test.iloc[i].to_dict(), top_n=10) for i in range(len(X_test))]


# Function to calculate Precision@k
def precision_at_k(true_labels, predictions, k):
    precision_scores = []
    for true, pred in zip(true_labels, predictions):
        true_set = set(true)
        pred_set = set(pred[:k])
        precision = len(true_set & pred_set) / k
        precision_scores.append(precision)
    return np.mean(precision_scores)

# Map test data to its true cluster group
test_true_labels = [data[data.index == idx]['cluster_label'].iloc[0] for idx in X_test.index]
test_true_labels = [[idx for idx in data[data['cluster_label'] == label].index] for label in test_true_labels]

# Calculate Precision@5
precision_score = precision_at_k(test_true_labels, predictions, 5)
print(f"Precision@5: {precision_score}")
