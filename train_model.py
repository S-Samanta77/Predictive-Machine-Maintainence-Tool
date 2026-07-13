import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
import joblib

# LOAD DATASET

data = pd.read_csv(r"D:\SOHAM\CODES\PYTHON WORKSPACE\datasets\ai4i2020.csv")

# BASIC EDA

print(data.shape)
print(data.head())
print(data.info())
print(data.describe())

print(data.isnull().sum())

# DROP UNNECESSARY COLUMNS

data.drop(["UDI","Product ID"],axis=1,inplace=True)

# ENCODE PRODUCT TYPE

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["Type"] = encoder.fit_transform(data["Type"])

# CORRELATION

plt.figure(figsize=(12,8))
sns.heatmap(data.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.tight_layout()
plt.show()

# FEATURE SELECTION

X = data.drop([
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
],axis=1)

Y = data["Machine failure"]

print("\nFeatures Used:")
print(X.columns)

print("\nTarget:")
print(Y.name)

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("\nTraining Shape :",X_train.shape)
print("Testing Shape :",X_test.shape)

# IMPORT METRICS

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, Y_train)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:,1]

print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

print("Accuracy :", accuracy_score(Y_test, lr_pred))
print("Precision:", precision_score(Y_test, lr_pred))
print("Recall   :", recall_score(Y_test, lr_pred))
print("F1 Score :", f1_score(Y_test, lr_pred))
print("ROC-AUC  :", roc_auc_score(Y_test, lr_prob))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test, lr_pred))

print("\nClassification Report")
print(classification_report(Y_test, lr_pred))

# DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=42)

parameter = {

    "criterion":[
        "gini",
        "entropy"
    ],

    "max_depth":[
        None,
        5,
        25,
        30,
        40
    ],

    "min_samples_split":[
        2,
        5,
        7
    ],

    "min_samples_leaf":[
        1,
        2
    ],

    "max_leaf_nodes": [140],

    "ccp_alpha": [0.00087],

    "class_weight":[
        None,
        "balanced"
    ]
}

grid_search = GridSearchCV(

    estimator=dt,

    param_grid=parameter,

    cv=5,

    scoring="f1",

    n_jobs=-1

)

grid_search.fit(X_train,Y_train)

grid_pred = grid_search.predict(X_test)
grid_prob = grid_search.predict_proba(X_test)[:,1]

print("\n")
print("="*60)
print("DECISION TREE CLASSIFIER")
print("="*60)

print("Best Parameters")
print(grid_search.best_params_)

print()

print("Accuracy :",accuracy_score(Y_test,grid_pred))
print("Precision:",precision_score(Y_test,grid_pred))
print("Recall   :",recall_score(Y_test,grid_pred))
print("F1 Score :",f1_score(Y_test,grid_pred))
print("ROC-AUC  :",roc_auc_score(Y_test,grid_prob))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test,grid_pred))

print("\nClassification Report")
print(classification_report(Y_test,grid_pred))

# RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state=42)

random_parameter = {

    "n_estimators":[
        100,
        200,
        337
    ],

    "max_depth":[
        None,
        10,
        27
    ],

    "min_samples_split":[
        2,
        5,
        10
    ],

    "min_samples_leaf":[
        1,
        2,
        4
    ],

    "max_features":[
        "sqrt",
        "log2",
        0.6,
        None
    ],

    "bootstrap":[
        True,
        False
    ],

    "class_weight":[
        None,
        "balanced"
    ]

}

random = RandomizedSearchCV(

    estimator=rf,

    param_distributions=random_parameter,

    n_iter=30,

    cv=5,

    scoring="f1",

    random_state=42,

    n_jobs=-1

)

random.fit(X_train,Y_train)

rf_pred = random.predict(X_test)
rf_prob = random.predict_proba(X_test)[:,1]

print("\n")
print("="*60)
print("RANDOM FOREST CLASSIFIER")
print("="*60)

print("Best Parameters")
print(random.best_params_)

print()

print("Accuracy :",accuracy_score(Y_test,rf_pred))
print("Precision:",precision_score(Y_test,rf_pred))
print("Recall   :",recall_score(Y_test,rf_pred))
print("F1 Score :",f1_score(Y_test,rf_pred))
print("ROC-AUC  :",roc_auc_score(Y_test,rf_prob))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test,rf_pred))

print("\nClassification Report")
print(classification_report(Y_test,rf_pred))

# FEATURE IMPORTANCE

print("\n")
print("="*60)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*60)

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":random.best_estimator_.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance)

joblib.dump(importance, "importance.pkl")

# Optional Visualization

plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# MODEL COMPARISON

comparison = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Decision Tree",

        "Random Forest"

    ],

    "Accuracy":[

        accuracy_score(Y_test,lr_pred),

        accuracy_score(Y_test,grid_pred),

        accuracy_score(Y_test,rf_pred)

    ],

    "Precision":[

        precision_score(Y_test,lr_pred),

        precision_score(Y_test,grid_pred),

        precision_score(Y_test,rf_pred)

    ],

    "Recall":[

        recall_score(Y_test,lr_pred),

        recall_score(Y_test,grid_pred),

        recall_score(Y_test,rf_pred)

    ],

    "F1 Score":[

        f1_score(Y_test,lr_pred),

        f1_score(Y_test,grid_pred),

        f1_score(Y_test,rf_pred)

    ],

    "ROC-AUC":[

        roc_auc_score(Y_test,lr_prob),

        roc_auc_score(Y_test,grid_prob),

        roc_auc_score(Y_test,rf_prob)

    ]

})

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(comparison)

# SAVE LABEL ENCODER

# Save the label encoder so Streamlit can encode
# Product Type (L, M, H) exactly the same way.

joblib.dump(encoder, "label_encoder.pkl")

# SAVE FEATURE NAMES

feature_names = X.columns.tolist()

joblib.dump(feature_names, "feature_names.pkl")

# SAVE TRAINED MODEL

# Logistic Regression
# joblib.dump(lr, "logistic_regression.pkl")

# Decision Tree
# joblib.dump(grid_search.best_estimator_, "decision_tree.pkl")

# Random Forest
joblib.dump(random.best_estimator_, "random_forest.pkl")

# DISPLAY TOP IMPORTANT FEATURES

print("\n")
print("="*60)
print("TOP IMPORTANT FEATURES")
print("="*60)

top_features = importance.head(4)

print(top_features)

