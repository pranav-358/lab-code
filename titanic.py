"""Titanic Dataset - AIML Practical 6 to 11
PR6 predicts Fare using passenger class (Pclass). PR7–PR11 use the supplied Survived column. The uploaded file contains the target column.

Keep this .py file in the same folder as the dataset file.
Run: python AIML_PR6-11_Titanic.py
"""
# AIML Practicals 6-11 - Titanic Dataset
# Keep this Python file in the same folder as its dataset.
# Run from top to bottom. Install: pandas numpy matplotlib scikit-learn scipy

# STEP 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from scipy.cluster.hierarchy import dendrogram, linkage

# STEP 2: Load dataset
df = pd.read_csv("titanic.csv")
print(df.head())
print("Dataset shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# STEP 3: Prepare data for classification practicals
df = df.drop_duplicates().dropna(subset=["Survived"])
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
y = df["Survived"].astype(int)
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Classification target: Survived")
print("Number of features:", X.shape[1])
print(y.value_counts())

# ============================================================
# PR6: LINEAR REGRESSION AND POLYNOMIAL REGRESSION
# ============================================================
print('\n========== PR6: REGRESSION ==========')
reg_data = df[["Pclass", "Fare"]].dropna()
X_reg = reg_data[["Pclass"]]
y_reg = reg_data["Fare"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(Xr_train, yr_train)
linear_pred = linear_model.predict(Xr_test)
print('Linear Regression')
print('MSE:', mean_squared_error(yr_test, linear_pred))
print('MAE:', mean_absolute_error(yr_test, linear_pred))
print('R2 Score:', r2_score(yr_test, linear_pred))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
Xr_train_poly = poly.fit_transform(Xr_train)
Xr_test_poly = poly.transform(Xr_test)
poly_model = LinearRegression()
poly_model.fit(Xr_train_poly, yr_train)
poly_pred = poly_model.predict(Xr_test_poly)
print('\nPolynomial Regression')
print('MSE:', mean_squared_error(yr_test, poly_pred))
print('MAE:', mean_absolute_error(yr_test, poly_pred))
print('R2 Score:', r2_score(yr_test, poly_pred))

# ============================================================
# PR7: LOGISTIC REGRESSION AND KNN
# ============================================================
print('\n========== PR7: LOGISTIC REGRESSION AND KNN ==========')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
print('Logistic Regression')
print('Accuracy:', accuracy_score(y_test, log_pred))
print('Precision:', precision_score(y_test, log_pred, zero_division=0))
print('Recall:', recall_score(y_test, log_pred, zero_division=0))
print('F1 Score:', f1_score(y_test, log_pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, log_pred))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
print('\nKNN')
print('Accuracy:', accuracy_score(y_test, knn_pred))
print('Precision:', precision_score(y_test, knn_pred, zero_division=0))
print('Recall:', recall_score(y_test, knn_pred, zero_division=0))
print('F1 Score:', f1_score(y_test, knn_pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, knn_pred))

# ============================================================
# PR8: DECISION TREE AND NAIVE BAYES
# ============================================================
print('\n========== PR8: DECISION TREE AND NAIVE BAYES ==========')
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print('Decision Tree')
print('Accuracy:', accuracy_score(y_test, tree_pred))
print('Precision:', precision_score(y_test, tree_pred, zero_division=0))
print('Recall:', recall_score(y_test, tree_pred, zero_division=0))
print('F1 Score:', f1_score(y_test, tree_pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, tree_pred))

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print('\nNaive Bayes')
print('Accuracy:', accuracy_score(y_test, nb_pred))
print('Precision:', precision_score(y_test, nb_pred, zero_division=0))
print('Recall:', recall_score(y_test, nb_pred, zero_division=0))
print('F1 Score:', f1_score(y_test, nb_pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, nb_pred))

# ============================================================
# PR9: K-MEANS AND HIERARCHICAL CLUSTERING
# ============================================================
print('\n========== PR9: CLUSTERING ==========')
# Use up to 1,000 rows for clustering so it runs quickly
X_cluster = X_train_scaled[:min(1000, len(X_train_scaled))]
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)
print('K-Means cluster counts:')
print(pd.Series(kmeans_labels).value_counts().sort_index())

hierarchical = AgglomerativeClustering(n_clusters=3)
hc_labels = hierarchical.fit_predict(X_cluster)
print('Hierarchical cluster counts:')
print(pd.Series(hc_labels).value_counts().sort_index())

# Dendrogram on up to 100 rows to keep it readable and fast
sample_size = min(100, len(X_cluster))
linked = linkage(X_cluster[:sample_size], method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Show clusters using the first two input features
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=kmeans_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clusters')
plt.show()
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=hc_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clusters')
plt.show()

# ============================================================
# PR10: PCA AND CLASSIFICATION COMPARISON
# ============================================================
print('\n========== PR10: PCA ==========')
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print('Features before PCA:', X_train_scaled.shape[1])
print('Features after PCA:', X_train_pca.shape[1])
print('Explained variance:', round(sum(pca.explained_variance_ratio_) * 100, 2), '%')
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()

original_model = LogisticRegression(max_iter=1000)
original_model.fit(X_train_scaled, y_train)
original_pred = original_model.predict(X_test_scaled)
pca_model = LogisticRegression(max_iter=1000)
pca_model.fit(X_train_pca, y_train)
pca_pred = pca_model.predict(X_test_pca)
print('Original feature accuracy:', accuracy_score(y_test, original_pred))
print('PCA feature accuracy:', accuracy_score(y_test, pca_pred))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_ * 100)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('PCA Explained Variance')
plt.show()

# ============================================================
# PR11: SVM, RANDOM FOREST, ADABOOST, CROSS VALIDATION, GRID SEARCH
# ============================================================
print('\n========== PR11: GRID SEARCH AND CROSS VALIDATION ==========')

# SVM uses scaled data
svm_grid = GridSearchCV(SVC(), {'C': [0.1, 1, 10]}, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
print('SVM best parameters:', svm_grid.best_params_)
print('SVM best CV score:', svm_grid.best_score_)
print('SVM test accuracy:', accuracy_score(y_test, svm_grid.predict(X_test_scaled)))

# Random Forest does not need feature scaling
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [50, 100], 'max_depth': [None, 5]},
    cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
print('\nRandom Forest best parameters:', rf_grid.best_params_)
print('Random Forest best CV score:', rf_grid.best_score_)
print('Random Forest test accuracy:', accuracy_score(y_test, rf_grid.predict(X_test)))

# AdaBoost does not need feature scaling
ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=42),
    {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]},
    cv=5, scoring='accuracy')
ada_grid.fit(X_train, y_train)
print('\nAdaBoost best parameters:', ada_grid.best_params_)
print('AdaBoost best CV score:', ada_grid.best_score_)
print('AdaBoost test accuracy:', accuracy_score(y_test, ada_grid.predict(X_test)))
