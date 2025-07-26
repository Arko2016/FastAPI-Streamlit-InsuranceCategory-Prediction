## What does sklearn.pipeline.Pipeline do?
A scikit-learn Pipeline provides a way to chain together multiple data processing steps and a final estimator into a single, cohesive unit. This simplifies the workflow for machine learning tasks, ensuring consistency and preventing data leakage

Example:
```python
# Define the pipeline steps*
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Scale the features
    ('pca', PCA(n_components=2)),      # Step 2: Apply Principal Component Analysis
    ('classifier', LogisticRegression()) # Step 3: Train a Logistic Regression model
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)
```

## What is the benefit of sklearn.compose.ColumnTransformer
It allows for applying different data transformations to different subsets of columns in a dataset. This is particularly useful when dealing with heterogeneous data, where some columns might be numerical and require scaling, while others are categorical and need encoding

ColumnTransformer takes a list of "transformers," where each transformer is a tuple consisting of:
1. A name (string): A unique identifier for the transformer.
2. A transformer object: An instance of a scikit-learn transformer (e.g., StandardScaler, OneHotEncoder, SimpleImputer)
3. Column(s): A list of column names or indices to which the transformer should be applied.

The ColumnTransformer then applies each specified transformer to its respective columns and concatenates the results horizontally, forming a single transformed feature space

Example:
```python
# Define transformers for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'fare']),  # Apply StandardScaler to numerical columns
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'embarked']) # Apply OneHotEncoder to categorical columns
    ],
    remainder='passthrough' # Keep other columns as they are (if any)
)

# Apply the transformations
transformed_data = preprocessor.fit_transform(df)
```


