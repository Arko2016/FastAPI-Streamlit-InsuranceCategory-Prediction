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

