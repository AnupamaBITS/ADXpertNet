import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# Load CSV dataset
data = pd.read_csv('MilkBinarySpectraDataset.csv')

# Handle missing values
if data.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='mean')  
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# Extract features and labels
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values   
print(y.shape)
sm = SMOTE(random_state=42, sampling_strategy = 'auto', k_neighbors=5)
X_res, y_res = sm.fit_resample(X,y)
print(y_res.shape)   


# Convert NumPy arrays to Pandas DataFrame
X_res_df = pd.DataFrame(X_res, columns=data.columns[:-1])  # Keep original feature names
y_res_series = pd.Series(y_res, name='Ingredient')  # Rename column appropriately

# Concatenate into a single DataFrame
data = pd.concat([X_res_df, y_res_series], axis=1)

print(data)

X = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values  # First 12 columns
y = data.iloc[:, -1].astype('category').cat.codes.values  # Spectra columns

# Encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  

# Print label encoding mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel Encoding Mapping:")
for class_name, encoded_value in label_mapping.items():
    print(f"{class_name}: {encoded_value}")

# Standardize feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('X',X_scaled)
print('y',y_encoded)
# Define models
# models = {
#     "Naive Bayes": GaussianNB(var_smoothing=1e-8),
#     "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_split=10),
#     "Linear SVM": LinearSVC(C=1, max_iter=100),
#     "SVM (RBF Kernel)": SVC(C=2.0, kernel='rbf', gamma='scale'),
#     "Logistic Regression": LogisticRegression(
#         penalty='l2', C=0.1, solver='lbfgs', max_iter=500, tol=1e-5,
#         class_weight='balanced'
#     ),
#      "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
# }

models = {
   "Naive Bayes": GaussianNB(var_smoothing=2.88),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=19),
    "SVM (Linear)": LinearSVC(C=0.1, max_iter=1000, dual=False),
    "SVM (RBF)": SVC(C=10.0, kernel='rbf', gamma=0.00186),
    "Logistic Regression": LogisticRegression(penalty='l2', C=10, solver='saga', max_iter=1000, tol=1e-5, class_weight='balanced', multi_class='multinomial'),
    "XGBoost": XGBClassifier(colsample_bytree=0.5,max_depth=10, n_estimators=334, learning_rate=0.21, subsample=0.5, reg_alpha=0, reg_lambda=0, eval_metric='mlogloss', use_label_encoder=False)
}

# Store results
results = {}

# Train and evaluate each model for 10 iterations
for model_name, model in models.items():
    accuracies, f1_scores, precisions, recalls, specificities = [], [], [], [], []
    class_wise_accuracies = np.zeros((10, len(label_encoder.classes_)))  # Store class-wise accuracy for each run

    print(f"\nTraining {model_name} for 10 iterations...")

    for i in range(10):
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=i)


        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro')

        # Compute specificity for each class
        cm = confusion_matrix(y_test, y_pred)
        specificity_per_class = []
        for j in range(len(cm)):
            tn = cm.sum() - (cm[j, :].sum() + cm[:, j].sum() - cm[j, j])
            fp = cm[:, j].sum() - cm[j, j]
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            specificity_per_class.append(specificity)

        specificities.append(np.mean(specificity_per_class))

        # Compute class-wise accuracy
        class_wise_acc = cm.diagonal() / cm.sum(axis=1)
        class_wise_acc = np.nan_to_num(class_wise_acc)  # Handle division by zero
        class_wise_accuracies[i] = class_wise_acc

        # Store scores
        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Store mean and std results (converted to %)
    results[model_name] = {
        "Mean Accuracy": np.mean(accuracies),
        "Std Accuracy": np.std(accuracies),
        "F1 Score": np.mean(f1_scores),
        "Std F1": np.std(f1_scores),
        "Precision": np.mean(precisions),
        "Std Precision": np.std(precisions),
        "Recall": np.mean(recalls),
        "Std Recall": np.std(recalls),
        "Specificity": np.mean(specificities),
        "Std Specificity": np.std(specificities),
        "Class-Wise Accuracy": np.mean(class_wise_accuracies, axis=0),
        "Std class-Wise Accuracy": np.std(class_wise_accuracies, axis=0)
    }

# Identify best model based on Mean Accuracy
best_model = max(results, key=lambda k: results[k]["Mean Accuracy"])

# Print results
print("\n---------------------------------------")
print(" Model Evaluation Summary")
print("---------------------------------------")
for model_name, metrics in results.items():
    print(f"\n🔹 {model_name}")
    print(f"   - Mean Accuracy: {metrics['Mean Accuracy']*100:.2f}% ± {metrics['Std Accuracy']*100:.2f}%")
    print(f"   - F1 Score: {metrics['F1 Score']*100:.2f}% ± {metrics['Std F1']*100:.2f}%")
    print(f"   - Precision: {metrics['Precision']*100:.2f}% ± {metrics['Std Precision']*100:.2f}%")
    print(f"   - Recall: {metrics['Recall']*100:.2f}% ± {metrics['Std Recall']*100:.2f}%")
    print(f"   - Specificity: {metrics['Specificity']*100:.2f}% ± {metrics['Std Specificity']*100:.2f}%")
    print("   - Class-Wise Accuracy (Mean ± Std):")
    for i, (mean_acc, std_acc) in enumerate(zip(metrics["Class-Wise Accuracy"], metrics["Std class-Wise Accuracy"])):
        print(f"     • Class {i}: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")


# Print best model
print("\n---------------------------------------")
print(f" ✅ Best Model: {best_model} with {results[best_model]['Mean Accuracy']*100:.2f}% accuracy ± {metrics['Std Accuracy']*100:.2f}%")
