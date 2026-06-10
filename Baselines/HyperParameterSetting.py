import time
import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from sklearn.model_selection import train_test_split, cross_val_score

# === Load and preprocess dataset ===
data = pd.read_csv('MilkSpectraDataset1.csv')
X = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
y = data.iloc[:, -1].astype('category').cat.codes.values

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print label encoding mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel Encoding Mapping:")
for class_name, encoded_value in label_mapping.items():
    print(f"{class_name}: {encoded_value}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42, shuffle=True)

# === Model list ===
model_list = [
    "Naive Bayes",
    "Decision Tree",
    "SVM (Linear)",
    "SVM (RBF)",
    "Logistic Regression",
    "XGBoost"
]

# === Loop through all models ===
for model_name in model_list:
    print(f"\n--- Optimizing {model_name} ---")

    def objective(trial):
        if model_name == "Naive Bayes":
            var_smoothing = trial.suggest_float("var_smoothing", 1e-9, 10.0, log=True)
            model = GaussianNB(var_smoothing=var_smoothing)

        elif model_name == "Decision Tree":
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

        elif model_name == "SVM (Linear)":
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            model = make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=1000, dual=False))

        elif model_name == "SVM (RBF)":
            C = trial.suggest_float("C", 0.1, 100.0, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
            model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel='rbf'))

        elif model_name == "Logistic Regression":
            C = trial.suggest_float("C", 0.01, 100.0, log=True)
            tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)
            model = make_pipeline(StandardScaler(), LogisticRegression(
                penalty='l2', C=C, solver='saga', max_iter=1000, tol=tol,
                class_weight='balanced', multi_class='multinomial'))

        elif model_name == "XGBoost":
            max_depth = trial.suggest_int("max_depth", 3, 15)
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
            subsample = trial.suggest_float("subsample", 0.3, 1.0)
            reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)

            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                colsample_bytree=colsample_bytree,
                subsample=subsample,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda
            )

        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        return score

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    # Best result
    print(f"\nBest hyperparameters for {model_name}:")
    print(study.best_params)
    print(f"Best cross-validated accuracy: {study.best_value:.4f}")

    # Final evaluation on test set
    if model_name == "SVM (Linear)":
        final_model = make_pipeline(StandardScaler(), LinearSVC(**study.best_params, max_iter=1000, dual=False))
    elif model_name == "SVM (RBF)":
        final_model = make_pipeline(StandardScaler(), SVC(**study.best_params, kernel='rbf'))
    elif model_name == "Logistic Regression":
        final_model = make_pipeline(StandardScaler(), LogisticRegression(**study.best_params,
                                penalty='l2', solver='saga', max_iter=1000,
                                class_weight='balanced', multi_class='multinomial'))
    elif model_name == "XGBoost":
        final_model = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss')
    elif model_name == "Naive Bayes":
        final_model = GaussianNB(**study.best_params)
    elif model_name == "Decision Tree":
        final_model = DecisionTreeClassifier(**study.best_params)

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test set accuracy for {model_name}: {test_acc:.4f}")
