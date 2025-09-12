import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datasets import load_dataset

# Load the dataset from Hugging Face Hub
HF_USERNAME = "sudha1726"  # Replace with your Hugging Face username
HF_DATASET_REPO = "tourism-package-purchase-dataset"
dataset = load_dataset(f"{HF_USERNAME}/{HF_DATASET_REPO}")

# Convert to pandas DataFrame
df = dataset['train'].to_pandas()


print("Dataset loaded successfully from Hugging Face Hub.")
display(df.head())

# Check for missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

# Handle missing values
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'ProdTaken' and 'CustomerID' from numerical columns for median imputation
if 'ProdTaken' in numerical_cols:
    numerical_cols.remove('ProdTaken')
if 'CustomerID' in numerical_cols:
    numerical_cols.remove('CustomerID')
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Drop rows with NaN in the target variable 'ProdTaken'
df.dropna(subset=['ProdTaken'], inplace=True)

# Separate target variable
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Identify categorical and numerical features for encoding
# Exclude 'CustomerID' from features
if 'CustomerID' in X.columns:
    X = X.drop('CustomerID', axis=1)
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('scaler', 'passthrough') # No scaling needed for this model for now, but keeping the structure
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print("\nData preparation complete. Data split into training and testing sets.")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

#upload the resulting train and test datasets to hugging space dataset
from datasets import Dataset
from huggingface_hub import HfApi
import pandas as pd

# Convert the processed data back to pandas DataFrames (if necessary for easier conversion to Dataset)
# Note: X_train and X_test are sparse matrices after one-hot encoding, so convert them to dense arrays or DataFrames
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# Combine features and target for creating datasets
train_df = pd.concat([X_train_df, pd.DataFrame(y_train.reset_index(drop=True), columns=['ProdTaken'])], axis=1)
test_df = pd.concat([X_test_df, pd.DataFrame(y_test.reset_index(drop=True), columns=['ProdTaken'])], axis=1)


# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Define the dataset repository on Hugging Face Hub
HF_USERNAME = "sudha1726"  # Replace with your Hugging Face username
HF_DATASET_REPO = "tourism-package-purchase-dataset" # Use the same repo or a new one

# Get your Hugging Face token from Colab secrets
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# Push the datasets to the Hugging Face Hub
train_dataset.push_to_hub(f"{HF_USERNAME}/{HF_DATASET_REPO}", split="train", token=HF_TOKEN)
test_dataset.push_to_hub(f"{HF_USERNAME}/{HF_DATASET_REPO}", split="test", token=HF_TOKEN)

print(f"Processed training and testing datasets uploaded to https://huggingface.co/datasets/{HF_USERNAME}/{HF_DATASET_REPO}")
