"""
PR5: Data Preprocessing and Exploratory Data Analysis (EDA)
Dataset: Titanic passenger dataset

Install required libraries (if not already installed):
    pip install pandas matplotlib seaborn scikit-learn

The program automatically downloads Titanic-Dataset.csv if it is not
available in the current working folder.
"""

import os
import urllib.request

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------- STEP 1: LOAD DATASET --------------------
DATASET_FILE = "Titanic-Dataset.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)

if not os.path.exists(DATASET_FILE):
    print("Dataset not found. Downloading Titanic dataset...")
    try:
        urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
        print("Dataset downloaded successfully.")
    except Exception as error:
        raise RuntimeError(
            "Could not download the Titanic dataset. "
            "Download the CSV from the dataset source and place "
            "'Titanic-Dataset.csv' in the same folder as this program. "
            f"Download error: {error}"
        )

df = pd.read_csv(DATASET_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nDataset shape:", df.shape)


# -------------------- STEP 2: DATA CLEANING --------------------
print("\n========== DATA CLEANING ==========")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

rows_before = len(df)
df = df.drop_duplicates().copy()
print("\nDuplicate rows removed:", rows_before - len(df))

# Remove columns that are not useful for this practical.
# Cabin has a large number of missing values in the standard dataset.
columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
df = df.drop(columns=columns_to_drop, errors="ignore")

# Remove rows where the target value is missing, if any.
df = df.dropna(subset=["Survived"]).copy()

print("\nColumns after cleaning:")
print(df.columns.tolist())


# -------------------- STEP 3: STATISTICAL ANALYSIS --------------------
print("\n========== STATISTICAL ANALYSIS ==========")

print("\nNumerical summary:")
print(df.describe())

print("\nCategorical summary:")
print(df.describe(include=["object"]))

print("\nSurvival count (0 = Not Survived, 1 = Survived):")
print(df["Survived"].value_counts())


# -------------------- STEP 4: DATA VISUALIZATION --------------------
print("\n========== DATA VISUALIZATION ==========")

numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Histograms show the distribution of numerical features.
df[numeric_columns].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Histograms of Titanic Dataset", fontsize=14)
plt.tight_layout()
plt.show()

# Boxplots help identify the spread and possible outliers.
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_columns])
plt.title("Boxplots of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------- STEP 5: SEPARATE FEATURES AND TARGET --------------------
print("\n========== FEATURE AND TARGET SEPARATION ==========")

X = df.drop("Survived", axis=1)
y = df["Survived"]

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)


# -------------------- STEP 6: TRAIN-TEST SPLIT --------------------
# Split before fitting imputers, encoders, or scalers to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# -------------------- STEP 7: MISSING VALUE HANDLING, ENCODING AND SCALING --------------------
print("\n========== DATA PREPROCESSING ==========")

# Numerical columns:
# 1. Replace missing values with the median.
# 2. Standardize values using StandardScaler.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical columns:
# 1. Replace missing values with the most frequent category.
# 2. Convert categories into numeric columns using One-Hot Encoding.
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

# Fit only on training data; apply the same transformations to test data.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert output to dense arrays when the transformer returns sparse matrices.
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)
X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

print("Missing values handled successfully.")
print("Categorical features encoded using One-Hot Encoding.")
print("Numerical features scaled using StandardScaler.")

print("\nProcessed training data (first five rows):")
print(X_train_processed.head())

print("\nProcessed testing data (first five rows):")
print(X_test_processed.head())

print("\nFinal training data shape:", X_train_processed.shape)
print("Final testing data shape:", X_test_processed.shape)

print("\nPR5: Data Preprocessing and EDA completed successfully.")
