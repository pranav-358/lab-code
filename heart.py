"""
AIML PRACTICALS 6–11
Dataset: CDC Heart Disease Statistics CSV stored inside "heart disease.zip"

IMPORTANT:
- Keep this Python file and "heart disease.zip" in the same folder.
- The ZIP must contain "heart disease.csv".
- Install required libraries:
    pip install pandas numpy matplotlib scikit-learn scipy
- Run:
    python AIML_PR6-11_Heart_Statistics_FINAL.py

Dataset note:
This CDC dataset contains population-level health statistics, not individual patient
records. PR6 predicts Data_Value from Year. PR7–PR11 classify whether Data_Value is
at or above the dataset-sample median. This is an educational target, NOT a medical
diagnosis.
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ============================================================
# STEP 2: LOAD AND PREPARE DATASET
# ============================================================

ZIP_FILE = "heart disease.zip"
CSV_FILE = "heart disease.csv"

print("\nLoading dataset...")

try:
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_file:
        with zip_file.open(CSV_FILE) as csv_file:
            # Read the first 50,000 records to keep execution manageable.
            df = pd.read_csv(csv_file, nrows=50000, low_memory=False)
except FileNotFoundError:
    raise SystemExit(
        f"\nERROR: Could not find '{ZIP_FILE}'. "
        "Place the ZIP file in the same folder as this Python program."
    )
except KeyError:
    raise SystemExit(
        f"\nERROR: '{CSV_FILE}' was not found inside '{ZIP_FILE}'. "
        "Check the CSV filename inside the ZIP."
    )

if "Data_Value" not in df.columns:
    raise SystemExit("ERROR: The dataset must contain a 'Data_Value' column.")

df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")
df = df.dropna(subset=["Data_Value"]).copy()

if len(df) > 5000:
    df = df.sample(n=5000, random_state=RANDOM_STATE).copy()

if len(df) < 20:
    raise SystemExit("ERROR: Not enough valid rows in the dataset to run the practicals.")

# Create a binary target for the classification practicals.
median_value = df["Data_Value"].median()
df["High_Value"] = (df["Data_Value"] >= median_value).astype(int)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nClassification target:")
print("0 = Data_Value below median")
print("1 = Data_Value at or above median")
print("Median Data_Value:", round(median_value, 3))
print("\nNOTE: This target is not a patient-level diagnosis.")


# ============================================================
# COMMON CLASSIFICATION DATA PREPARATION
# ============================================================

# Use available columns from the supplied dataset. Data_Value is deliberately
# excluded from the predictors because it is used to create the target.
candidate_features = [
    "Year",
    "LocationAbbr",
    "LocationDesc",
    "GeographicLevel",
    "Topic",
    "Stratification1",
    "Stratification2",
    "Stratification3",
    "LocationID"
]
features = [column for column in candidate_features if column in df.columns]

if not features:
    raise SystemExit("ERROR: No expected feature columns were found in the dataset.")

X = df[features].copy()
y = df["High_Value"].copy()

# Convert numeric-looking columns to numeric values.
for column in X.columns:
    if column in ["Year", "LocationID"]:
        X[column] = pd.to_numeric(X[column], errors="coerce")

# One-hot encode categorical features. This does not use the target values.
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True, dtype=float)
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain no usable values.
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise SystemExit("ERROR: No usable predictor columns remain after preprocessing.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Fit imputation and scaling using training data only to avoid test-set leakage.
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("\nClassification data prepared.")
print("Features used:", X.shape[1])
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classification_metrics(y_true, y_pred):
    """Return standard binary classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }


def print_classification_results(model_name, y_true, y_pred):
    """Print metrics and display a confusion matrix."""
    scores = classification_metrics(y_true, y_pred)

    print(f"\n{model_name}")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Value (0)", "High Value (1)"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return scores


# ============================================================
# PR6: REGRESSION TECHNIQUES
# Linear Regression and Polynomial Regression
# ============================================================

print("\n" + "=" * 65)
print("PR6: REGRESSION TECHNIQUES")
print("=" * 65)

# Predict Data_Value using Year.
reg_data = df[["Year", "Data_Value"]].dropna().copy()
reg_data["Year"] = pd.to_numeric(reg_data["Year"], errors="coerce")
reg_data = reg_data.dropna()

X_reg = reg_data[["Year"]]
y_reg = reg_data["Data_Value"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(Xr_train, yr_train)
linear_pred = linear_model.predict(Xr_test)

linear_scores = {
    "MSE": mean_squared_error(yr_test, linear_pred),
    "MAE": mean_absolute_error(yr_test, linear_pred),
    "R2 Score": r2_score(yr_test, linear_pred)
}

print("\nLINEAR REGRESSION")
for metric, value in linear_scores.items():
    print(f"{metric}: {value:.4f}")

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
Xr_train_poly = poly_features.fit_transform(Xr_train)
Xr_test_poly = poly_features.transform(Xr_test)

poly_model = LinearRegression()
poly_model.fit(Xr_train_poly, yr_train)
poly_pred = poly_model.predict(Xr_test_poly)

poly_scores = {
    "MSE": mean_squared_error(yr_test, poly_pred),
    "MAE": mean_absolute_error(yr_test, poly_pred),
    "R2 Score": r2_score(yr_test, poly_pred)
}

print("\nPOLYNOMIAL REGRESSION (DEGREE 2)")
for metric, value in poly_scores.items():
    print(f"{metric}: {value:.4f}")

print("\nREGRESSION COMPARISON")
regression_comparison = pd.DataFrame(
    [linear_scores, poly_scores],
    index=["Linear Regression", "Polynomial Regression"]
)
print(regression_comparison.round(4))
print("Lower MSE and MAE are better; higher R2 is better.")

# Plot actual values and both fitted models.
plot_years = np.linspace(X_reg["Year"].min(), X_reg["Year"].max(), 300).reshape(-1, 1)
linear_curve = linear_model.predict(plot_years)
poly_curve = poly_model.predict(poly_features.transform(plot_years))

plt.figure(figsize=(9, 5))
plt.scatter(Xr_test["Year"], yr_test, alpha=0.55, label="Actual test data")
plt.plot(plot_years.ravel(), linear_curve, label="Linear Regression")
plt.plot(plot_years.ravel(), poly_curve, label="Polynomial Regression (degree 2)")
plt.xlabel("Year")
plt.ylabel("Data_Value")
plt.title("PR6: Linear and Polynomial Regression")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ============================================================
# PR7: LOGISTIC REGRESSION AND K-NEAREST NEIGHBORS
# ============================================================

print("\n" + "=" * 65)
print("PR7: LOGISTIC REGRESSION AND KNN")
print("=" * 65)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
logistic_model.fit(X_train_scaled, y_train)
logistic_pred = logistic_model.predict(X_test_scaled)

logistic_scores = print_classification_results(
    "Logistic Regression", y_test, logistic_pred
)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

knn_scores = print_classification_results("K-Nearest Neighbors (KNN)", y_test, knn_pred)

print("\nPR7 MODEL COMPARISON")
pr7_comparison = pd.DataFrame(
    [logistic_scores, knn_scores],
    index=["Logistic Regression", "KNN"]
)
print(pr7_comparison.round(4))
print("Compare the models using F1-score, precision, recall and accuracy.")


# ============================================================
# PR8: DECISION TREE AND NAIVE BAYES
# ============================================================

print("\n" + "=" * 65)
print("PR8: DECISION TREE AND NAIVE BAYES")
print("=" * 65)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_model.fit(X_train_imputed, y_train)
tree_pred = tree_model.predict(X_test_imputed)

tree_scores = print_classification_results(
    "Decision Tree", y_test, tree_pred
)

# Gaussian Naive Bayes
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_imputed, y_train)
naive_bayes_pred = naive_bayes_model.predict(X_test_imputed)

naive_bayes_scores = print_classification_results(
    "Gaussian Naive Bayes", y_test, naive_bayes_pred
)

print("\nPR8 MODEL COMPARISON")
pr8_comparison = pd.DataFrame(
    [tree_scores, naive_bayes_scores],
    index=["Decision Tree", "Gaussian Naive Bayes"]
)
print(pr8_comparison.round(4))
print("The model with the stronger evaluation results is more suitable for this dataset.")


# ============================================================
# PR9: CLUSTERING TECHNIQUES
# K-Means and Hierarchical Clustering
# ============================================================

print("\n" + "=" * 65)
print("PR9: CLUSTERING TECHNIQUES")
print("=" * 65)

# Use a manageable subset of the scaled training data.
cluster_size = min(1000, len(X_train_scaled))
X_cluster = X_train_scaled[:cluster_size]

if cluster_size < 10:
    raise SystemExit("Not enough records for clustering.")

# Elbow Method: calculate WCSS for K = 1 to 10.
wcss = []
k_values = range(1, min(10, cluster_size - 1) + 1)

for k in k_values:
    kmeans_test = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    kmeans_test.fit(X_cluster)
    wcss.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.title("PR9: Elbow Method")
plt.xticks(list(k_values))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Select K after inspecting the Elbow graph.
# Change this value if the elbow in your graph suggests another K.
OPTIMAL_K = 3
OPTIMAL_K = min(OPTIMAL_K, cluster_size - 1)

# K-Means Clustering
kmeans_model = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=RANDOM_STATE,
    n_init=10
)
kmeans_labels = kmeans_model.fit_predict(X_cluster)

print(f"\nK-MEANS CLUSTER COUNTS (K={OPTIMAL_K})")
print(pd.Series(kmeans_labels).value_counts().sort_index())

# Hierarchical Clustering
hierarchical_model = AgglomerativeClustering(n_clusters=OPTIMAL_K)
hierarchical_labels = hierarchical_model.fit_predict(X_cluster)

print(f"\nHIERARCHICAL CLUSTER COUNTS (K={OPTIMAL_K})")
print(pd.Series(hierarchical_labels).value_counts().sort_index())

# Dendrogram using a smaller sample for readability.
dendrogram_size = min(100, cluster_size)
hierarchical_linkage = linkage(
    X_cluster[:dendrogram_size],
    method="ward"
)

plt.figure(figsize=(12, 5))
dendrogram(hierarchical_linkage)
plt.title("PR9: Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

# Reduce clustering data to two dimensions for clear visualization.
# PCA is used only to visualize the clusters, not to perform clustering.
cluster_pca = PCA(n_components=2)
X_cluster_2d = cluster_pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 5))
plt.scatter(
    X_cluster_2d[:, 0],
    X_cluster_2d[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PR9: K-Means Clusters (2D PCA Visualization)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    X_cluster_2d[:, 0],
    X_cluster_2d[:, 1],
    c=hierarchical_labels,
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PR9: Hierarchical Clusters (2D PCA Visualization)")
plt.tight_layout()
plt.show()

print("\nPR9 ANALYSIS")
print("K-Means groups records by minimizing within-cluster distances.")
print("Hierarchical clustering groups records using a hierarchy of merges.")
print("Use the Elbow graph and Dendrogram to judge a suitable number of clusters.")
print("The cluster labels are group identifiers; they do not represent disease diagnoses.")


# ============================================================
# PR10: DIMENSIONALITY REDUCTION USING PCA
# ============================================================

print("\n" + "=" * 65)
print("PR10: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 65)

# Fit PCA only on training data, then transform test data.
# Two components are retained for visualization and comparison.
pca_model = PCA(n_components=2)
X_train_pca = pca_model.fit_transform(X_train_scaled)
X_test_pca = pca_model.transform(X_test_scaled)

print("Number of features before PCA:", X_train_scaled.shape[1])
print("Number of features after PCA:", X_train_pca.shape[1])
print(
    "Total explained variance by two components:",
    round(np.sum(pca_model.explained_variance_ratio_) * 100, 2),
    "%"
)

# PCA visualization
plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.65
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PR10: PCA 2D Visualization")
plt.colorbar(scatter, label="High_Value class")
plt.tight_layout()
plt.show()

# Explained variance graph
plt.figure(figsize=(7, 5))
plt.bar(
    ["PC1", "PC2"],
    pca_model.explained_variance_ratio_ * 100
)
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance (%)")
plt.title("PR10: Explained Variance by Principal Component")
plt.tight_layout()
plt.show()

# Compare Logistic Regression before and after PCA.
original_pca_comparison_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)
original_pca_comparison_model.fit(X_train_scaled, y_train)
original_pca_pred = original_pca_comparison_model.predict(X_test_scaled)

reduced_pca_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)
reduced_pca_model.fit(X_train_pca, y_train)
reduced_pca_pred = reduced_pca_model.predict(X_test_pca)

original_accuracy = accuracy_score(y_test, original_pca_pred)
pca_accuracy = accuracy_score(y_test, reduced_pca_pred)

print("\nCLASSIFICATION PERFORMANCE COMPARISON")
print(f"Accuracy using original features: {original_accuracy:.4f}")
print(f"Accuracy using PCA features: {pca_accuracy:.4f}")

plt.figure(figsize=(7, 5))
plt.bar(
    ["Original Features", "After PCA"],
    [original_accuracy, pca_accuracy]
)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("PR10: Classification Accuracy Before and After PCA")
plt.tight_layout()
plt.show()


# ============================================================
# PR11: ADVANCED MACHINE LEARNING AND MODEL OPTIMIZATION
# SVM, Random Forest, AdaBoost, Cross Validation and Grid Search
# ============================================================

print("\n" + "=" * 65)
print("PR11: ADVANCED MACHINE LEARNING AND MODEL OPTIMIZATION")
print("=" * 65)

# Baseline models: evaluate before hyperparameter optimization.
baseline_models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE)
}

baseline_scores = {}
baseline_cv_scores = {}

print("\nBASELINE MODEL PERFORMANCE (BEFORE OPTIMIZATION)")

for model_name, model in baseline_models.items():
    if model_name == "SVM":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        cv_scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5,
            scoring="accuracy"
        )
    else:
        model.fit(X_train_imputed, y_train)
        predictions = model.predict(X_test_imputed)
        cv_scores = cross_val_score(
            model,
            X_train_imputed,
            y_train,
            cv=5,
            scoring="accuracy"
        )

    baseline_scores[model_name] = classification_metrics(y_test, predictions)
    baseline_cv_scores[model_name] = cv_scores.mean()

    print(f"\n{model_name}")
    print("Baseline test accuracy:", round(baseline_scores[model_name]["Accuracy"], 4))
    print("5-Fold Cross Validation Accuracy:", round(cv_scores.mean(), 4))
    print_classification_results(
        f"{model_name} - Baseline",
        y_test,
        predictions
    )

# Hyperparameter tuning using GridSearchCV with 5-fold cross-validation.
print("\nHYPERPARAMETER TUNING USING GRID SEARCH")

svm_grid = GridSearchCV(
    SVC(),
    param_grid={
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)
svm_optimized_pred = svm_grid.predict(X_test_scaled)

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid={
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_train_imputed, y_train)
rf_optimized_pred = rf_grid.predict(X_test_imputed)

ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=RANDOM_STATE),
    param_grid={
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1.0]
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
ada_grid.fit(X_train_imputed, y_train)
ada_optimized_pred = ada_grid.predict(X_test_imputed)

optimized_models = {
    "SVM": (svm_grid, svm_optimized_pred),
    "Random Forest": (rf_grid, rf_optimized_pred),
    "AdaBoost": (ada_grid, ada_optimized_pred)
}

optimized_scores = {}
optimized_cv_scores = {}

for model_name, (grid, predictions) in optimized_models.items():
    optimized_scores[model_name] = classification_metrics(y_test, predictions)
    optimized_cv_scores[model_name] = grid.best_score_

    print(f"\n{model_name}")
    print("Best Parameters:", grid.best_params_)
    print("Best 5-Fold CV Accuracy:", round(grid.best_score_, 4))
    print("Optimized Test Accuracy:", round(optimized_scores[model_name]["Accuracy"], 4))

    print_classification_results(
        f"{model_name} - Optimized",
        y_test,
        predictions
    )

# Before-versus-after optimization comparison.
comparison_rows = []

for model_name in baseline_models:
    comparison_rows.append({
        "Model": model_name,
        "Baseline Accuracy": baseline_scores[model_name]["Accuracy"],
        "Optimized Accuracy": optimized_scores[model_name]["Accuracy"],
        "Baseline CV Accuracy": baseline_cv_scores[model_name],
        "Optimized CV Accuracy": optimized_cv_scores[model_name]
    })

pr11_comparison = pd.DataFrame(comparison_rows).set_index("Model")

print("\nPR11: BEFORE AND AFTER OPTIMIZATION COMPARISON")
print(pr11_comparison.round(4))

# Accuracy comparison graph
labels = list(baseline_models.keys())
baseline_accuracies = [baseline_scores[name]["Accuracy"] for name in labels]
optimized_accuracies = [optimized_scores[name]["Accuracy"] for name in labels]

positions = np.arange(len(labels))
bar_width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(
    positions - bar_width / 2,
    baseline_accuracies,
    bar_width,
    label="Before Optimization"
)
plt.bar(
    positions + bar_width / 2,
    optimized_accuracies,
    bar_width,
    label="After Grid Search"
)
plt.xticks(positions, labels)
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.title("PR11: Model Performance Before and After Optimization")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ALL SIX PRACTICALS (PR6–PR11) COMPLETED")
print("=" * 65)
print("PR6  - Linear Regression and Polynomial Regression")
print("PR7  - Logistic Regression and K-Nearest Neighbors")
print("PR8  - Decision Tree and Gaussian Naive Bayes")
print("PR9  - K-Means and Hierarchical Clustering")
print("PR10 - Principal Component Analysis")
print("PR11 - SVM, Random Forest, AdaBoost, Cross Validation and Grid Search")
print("\nAll requested practical sections have executed.")
print("Refer to the printed metrics, comparison tables and graphs for analysis.")
print("\nReminder: This CDC dataset contains population-level statistics.")
print("The derived High_Value class is an educational label, not a diagnosis.")
