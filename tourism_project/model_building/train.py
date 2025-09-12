!pip install mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi, create_repo
import joblib
import os

# Define MLflow tracking URI (if using a remote server)
# mlflow.set_tracking_uri("YOUR_MLFLOW_TRACKING_URI")

# Set MLflow experiment name
mlflow.set_experiment("tourism-package-purchase-prediction")

# Load the processed datasets from Hugging Face Hub
HF_USERNAME = "sudha1726"  # Replace with your Hugging Face username
HF_DATASET_REPO = "tourism-package-purchase-dataset"
HF_MODEL_REPO = "tourism-package-purchase-model" # New repo for the model

train_dataset = load_dataset(f"{HF_USERNAME}/{HF_DATASET_REPO}", split="train")
test_dataset = load_dataset(f"{HF_USERNAME}/{HF_DATASET_REPO}", split="test")

# Convert datasets to pandas DataFrames
train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()

# Separate features and target variable
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Start an MLflow run for hyperparameter tuning
with mlflow.start_run(run_name="hyperparameter_tuning"):
    # Define the model
    model = LogisticRegression(random_state=42)

    # Define the parameters to tune
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear', 'lbfgs']
    }

    # Tune the model with defined parameters using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_f1_score = grid_search.best_score_

    # Log the tuned parameters to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score_cv", best_f1_score)

    print("Hyperparameter tuning complete.")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation F1 Score: {best_f1_score:.4f}")

# Start a new MLflow run for training with the best model
with mlflow.start_run(run_name="best_model_training"):
    # Train the best model on the full training data
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Evaluate the best model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log evaluation metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("\nBest model training and evaluation complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_model")

# Register the best model in the Hugging Face Model Hub
HF_TOKEN = userdata.get('HF_TOKEN') # Get Hugging Face token from Colab secrets

api = HfApi()

# Create the model repository on Hugging Face Hub
create_repo(repo_id=f"{HF_USERNAME}/{HF_MODEL_REPO}", repo_type="model", exist_ok=True, token=HF_TOKEN)

# Save the best model locally
model_filename = "best_logistic_regression_model.pkl"
joblib.dump(best_model, model_filename)

# Upload the model file to the Hugging Face Model Hub
api.upload_file(
    path_or_fileobj=model_filename,
    path_in_repo=model_filename,
    repo_id=f"{HF_USERNAME}/{HF_MODEL_REPO}",
    repo_type="model",
    token=HF_TOKEN,
)

print(f"\nBest model uploaded to https://huggingface.co/{HF_USERNAME}/{HF_MODEL_REPO}")

# Clean up the locally saved model file
os.remove(model_filename)
