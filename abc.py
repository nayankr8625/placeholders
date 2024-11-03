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




# TFIDf and Cosine Similiarity

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Step 1: Handle missing values and combine all categorical and numerical features into a single text representation
data_combined = data[categorical_cols].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1) + ' ' + \
                data[numerical_cols].fillna(0).astype(str).apply(lambda x: ' '.join(x), axis=1)

# Assign the final combined text to a new 'combined_text' column
data['combined_text'] = data_combined

# Step 2: Process categorical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_text'])

# Step 3: Handle NaN values in numerical data globally and scale numerical features
data[numerical_cols] = data[numerical_cols].fillna(0)
scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(data[numerical_cols])

# Define function for weighted similarity combining categorical and numerical features
def find_similar_entries_weighted(entry_data, top_n=5, weight_cat=0.7, weight_num=0.3):
    # Convert entry data to a similar format
    entry_text = ' '.join([str(entry_data.get(col, '')) for col in categorical_cols + numerical_cols])
    entry_vector_cat = tfidf_vectorizer.transform([entry_text])
    cat_similarity_scores = cosine_similarity(entry_vector_cat, tfidf_matrix).flatten()
    
    # Process numerical features with NaN values filled
    entry_numerical_df = pd.DataFrame([entry_data], columns=numerical_cols).fillna(0).astype(float)
    entry_vector_num = scaler.transform(entry_numerical_df)
    num_similarity_scores = cosine_similarity(entry_vector_num, scaled_numerical_features).flatten()
    
    # Combine similarity scores using weights for categorical and numerical components
    combined_similarity = (weight_cat * cat_similarity_scores) + (weight_num * num_similarity_scores)
    
    # Retrieve top_n similar entries by combined similarity score
    top_indices = combined_similarity.argsort()[-top_n:][::-1]
    top_scores = combined_similarity[top_indices]
    
    similar_entries = {
        data.index[idx]: round(score * 100, 2)
        for idx, score in zip(top_indices, top_scores)
    }
    
    return similar_entries

# Create a mixed new entry by sampling values from 5 random entries
random_rows = data.sample(5, random_state=42).to_dict(orient='records')
mixed_entry = {col: random.choice([row[col] for row in random_rows]) for col in categorical_cols + numerical_cols}

# Run the similarity function with the mixed entry
similar_entries_mixed = find_similar_entries_weighted(mixed_entry)

# Display similar entries and their similarity scores
for identifier, score in similar_entries_mixed.items():
    print(f"{identifier}: {score}% Similarity")


