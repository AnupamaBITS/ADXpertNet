import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from codecarbon import EmissionsTracker

import torch.optim as optim

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import torch.nn.functional as F
import math

import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For full reproducibility (PyTorch 1.8+)
    torch.use_deterministic_algorithms(True)

set_seed(42)



# ------------------------------
# 1. Gaussian Noise
# ------------------------------
def add_gaussian_noise(spectra, noise_std=0.01, prob=0.5):
    if random.random() < prob:
        noise = np.random.normal(0, noise_std, spectra.shape)
        spectra = spectra + noise
    return spectra


# ------------------------------
# 2. Random Spectral Shift / Broadening
# ------------------------------
def random_shift_broadening(spectra, max_shift=3, prob=0.5):
    if random.random() < prob:
        shift = np.random.randint(-max_shift, max_shift)
        spectra = np.roll(spectra, shift)

        # Simple broadening via smoothing kernel
        kernel_size = np.random.choice([3,5])
        kernel = np.ones(kernel_size) / kernel_size
        spectra = np.convolve(spectra, kernel, mode='same')

    return spectra


# ------------------------------
# 3. Mixup (Linear Interpolation)
# ------------------------------
def mixup_augmentation(X_ref, X_spec, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(X_spec))

    X_ref_mix = lam * X_ref + (1 - lam) * X_ref[index]
    X_spec_mix = lam * X_spec + (1 - lam) * X_spec[index]

    # For classification (hard labels)
    y_mix = y  # keeping original label for simplicity

    return X_ref_mix, X_spec_mix, y_mix



class SpectralCNNLSTM(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_sizes,stride):
        super(SpectralCNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, num_filters[0], kernel_sizes[0], stride[0])
        self.dropout1 = nn.Dropout(0.2)
        
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], kernel_sizes[1], stride[1])
        
        self.dropout2 = nn.Dropout(0.2)
        

        # Third convolutional layer 
        self.conv3 = nn.Conv1d(num_filters[1], num_filters[2], kernel_sizes[2], stride[2])
        
        self.dropout3 = nn.Dropout(0.2)  # Corrected: New dropout for conv3
        
        
        #First LSTM Layer
        self.lstm1 =  nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.dropout_lstm1 = nn.Dropout(0.2)

        #Second LSTM Layer
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)  # GRU layer with input size 167, hidden size 100
        self.dropout_lstm2 = nn.Dropout(0.2)
        

        # Flatten layer before FC
        self.flatten = nn.Flatten()
        
        # Fully connected layers after LSTM
        self.fc1 = nn.Linear(in_features=3968, out_features=500)
        
        self.dropout_fc1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(500, 300)
        
        self.dropout_fc2 = nn.Dropout(0.2)
        
        # Output layer: 6 output classes
        self.fc3 = nn.Linear(300, 6)  # Final output layer
        
        # Softmax layer for classification
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        # x = self.batchnorm1(x)
        x = self.dropout1(x)
        #x = self.maxpool1(x)
        
        # Second conv block
        x = self.conv2(x)
        # x = self.batchnorm2(x)
        x = self.dropout2(x)
        #x = self.maxpool1(x)

        # Third conv block 
        x = self.conv3(x)  
        # x = self.batchnorm3(x)  
        x = self.dropout3(x)  
        #x = self.maxpool1(x)

        # Reshape LSTM (batch_size, seq_len, feature_size)
        x = x.permute(0, 2, 1)  # Rearrange to (batch_size, seq_len, features)
        
        # First LSTM block
        x, _ = self.lstm1(x)  # GRU layer output
        x = self.dropout_lstm1(x)
        x = x.permute(0, 2, 1)  # Rearrange back for MaxPool
        
        
        # Second LSTM block
        x = x.permute(0, 2, 1)  # Rearrange again for GRU
        x, _ = self.lstm2(x)  # GRU layer output
        x = self.dropout_lstm2(x)
        x = x.permute(0, 2, 1)  # Rearrange back for MaxPool
        
        
        # Flatten the output for FC layers
        x = self.flatten(x)
        
        # First fully connected block
        x = self.fc1(x)
        # x = self.batchnorm_fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout_fc1(x)
        
        # Second fully connected block
        x = self.fc2(x)
        # x = self.batchnorm_fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout_fc2(x)

        # Output layer (fully connected)
        x = self.fc3(x)
        
        # # Apply softmax for multiclass classification
        x = self.softmax(x)

        return x
    
# Function to compute additional metrics
def compute_metrics(y_true, y_pred, num_classes):
    # Precision, Recall, F1-Score
    precision = precision_score(y_true, y_pred, average=None, zero_division=1)
    recall = recall_score(y_true, y_pred, average=None, zero_division=1)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    specificity = []
    class_accuracy = []
    for i in range(num_classes):
        TN = cm.sum() - cm[:, i].sum() - cm[i, :].sum() + cm[i, i]
        FP = cm[:, i].sum() - cm[i, i]
        FN = cm[i, :].sum() - cm[i, i]
        TP = cm[i, i]
        specificity.append(TN / (TN + FP) if (TN + FP) != 0 else 0)
        class_accuracy.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
    
    return precision, recall, f1, specificity, class_accuracy


# def reset_weights(m):
#     if hasattr(m, 'reset_parameters'):
#         m.reset_parameters()


# Training Loop
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epoch=100):
    model.to(device)
    
    num_classes = 6  # Number of classes    
    num_iterations = 10

    # Initialize lists to store metrics across iterations
    all_accuracies = []
    all_f1 = []
    all_recall = []
    all_precision = []
    all_specificity = []
    all_class_accuracies = []  # To store class-wise accuracy
    best_metrics = {
        'accuracy': 0,
        'f1': 0,
        'recall': 0,
        'precision': 0,
        'specificity': 0
    }
    
    best_epoch_metrics = {}
    
    # Initialize iteration-wise metric storage
    for iteration in range(num_iterations):
        # model.apply(reset_weights)
        set_seed(42)
        print(f"\nTraining Iteration {iteration + 1}/{num_iterations}...")
        
        val_loss_history = []
        val_acc_history = []
        
        for epoch_num in range(epoch):
            model.train()
            
            # Training loop
            for reference, spectra, labels in train_loader:
                reference, spectra = reference.to(device), spectra.to(device)
                labels = labels.to(device)
                #labels = labels.argmax(dim=1)
                #print('shape of labels', labels.shape)
                # Forward pass
                outputs = model(spectra)
                loss = criterion(outputs, labels)
                
                
#                outputs = model(reference, spectra)
#                loss = criterion(outputs, labels)

                # outputs, align_loss = model(reference, spectra)
                # classification_loss = criterion(outputs, labels)
                # loss = classification_loss + model.align_weight * align_loss

                # outputs, semantic_loss, feature_loss = model(reference, spectra)

                # classification_loss = criterion(outputs, labels)

                # loss = classification_loss \
                #     + model.semantic_weight * semantic_loss \
                #     + model.feature_weight * feature_loss
                
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for reference, spectra, labels in val_loader:
                    reference, spectra = reference.to(device), spectra.to(device)
                    labels = labels.to(device)
                    #labels = labels.argmax(dim=1)
                    outputs = model(spectra)
                    loss = criterion(outputs, labels)
                    
                    
#                    outputs = model(reference, spectra)
#                    loss = criterion(outputs, labels)
                    # outputs, _, _ = model(reference, spectra)
                    # loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    #predicted = torch.sigmoid(outputs) >= 0.4  # Binary threshold
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            # Test set evaluation 
            model.eval()
            test_loss = 0.0
            correct_test, total_test = 0, 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for reference, spectra, labels in test_loader:
                    reference, spectra = reference.to(device), spectra.to(device)
                    labels = labels.to(device)
                    #labels = labels.argmax(dim=1)
                    outputs = model(spectra)
                    loss = criterion(outputs, labels)

                    # outputs, _ = model(reference, spectra)
                    # loss = criterion(outputs, labels)

                    # outputs, _, _ = model(reference, spectra)
                    # loss = criterion(outputs, labels)


                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    #print('predicted',predicted)
                    #predicted = (torch.sigmoid(outputs) >= 0.4).float().cpu().numpy()  # Binary threshold
                    
                    correct_test += (predicted == labels).sum().item()
                    total_test += labels.size(0)
                    
                    all_targets.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    

                    y_true = np.array(all_targets)
                    y_pred = np.array(all_preds)
                    # print('y_true shape',y_true)
                    # print('y_pred shape',y_pred)
                # Calculate metrics
                acc = accuracy_score(y_true, y_pred)

                # MULTICLASS metrics with 'macro' or 'weighted'
                f1 = f1_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
                precision = precision_score(y_true, y_pred, average='macro')

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (6, 6):
                    specificity_list = []
                    class_acc = cm.diagonal() / cm.sum(axis=1)
                    
                    for i in range(num_classes):
                        TP = cm[i, i]
                        FP = cm[:, i].sum() - TP
                        FN = cm[i, :].sum() - TP
                        TN = cm.sum() - (TP + FP + FN)
                        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                        specificity_list.append(specificity)
                    
                    avg_specificity = np.mean(specificity_list)
                    print(f"Epoch {epoch_num+1}: Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Avg Specificity={avg_specificity:.4f}")
                    print("Class-wise Accuracy:")
                    for i, class_name in enumerate(class_names):
                        print(f"{class_name}: {class_acc[i]:.4f}")

                else:
                    avg_specificity = 0
                    class_acc = [0] * num_classes


                # Store metrics
                all_accuracies.append(acc)
                all_f1.append(f1)
                all_recall.append(recall)
                all_precision.append(precision)
                all_specificity.append(specificity)
                all_class_accuracies.append(class_acc)
                
                # Calculate standard deviation for each metric (plus-minus variation)
                std_accuracy = np.std(all_accuracies)
                std_f1 = np.std(all_f1)
                std_recall = np.std(all_recall)
                std_precision = np.std(all_precision)
                std_specificity = np.std(all_specificity)
                std_class_accuracies = np.std(all_class_accuracies, axis=0)

                # Track the best model for this iteration
                if acc > best_metrics['accuracy']:
                    best_metrics['accuracy'] = acc
                    best_metrics['f1'] = f1
                    best_metrics['recall'] = recall
                    best_metrics['precision'] = precision
                    best_metrics['specificity'] = specificity
                    best_epoch_metrics = {
                        'epoch': epoch_num+1,
                        'accuracy': acc,
                        'f1': f1,
                        'recall': recall,
                        'precision': precision,
                        'specificity': specificity,
                        'class_accuracy': class_acc
                    }
         

        
        print(f"\nBest Model for Iteration {iteration + 1}:")
        print(f"Epoch {best_epoch_metrics['epoch']}: acc={best_epoch_metrics['accuracy']:.4f} ± {std_accuracy:.4f}, "
              f"Precision={best_epoch_metrics['precision']:.4f} ± {std_precision:.4f}, Recall={best_epoch_metrics['recall']:.4f} ± {std_recall:.4f}, "
              f"F1={best_epoch_metrics['f1']:.4f}  ± {std_f1:.4f}, Specificity={best_epoch_metrics['specificity']:.4f} ± {std_specificity:.4f}")
        
        print("Class-wise Accuracy (Best Model):")
        for i, class_name in enumerate(class_names):
            mean_acc = best_epoch_metrics['class_accuracy'][i]
            std_acc = std_class_accuracies[i]
            print(f"{class_name}: {mean_acc:.4f} +/- {std_acc:.4f}")
    
    # ----------------------------
    # Stop Tracker
    # ----------------------------
    emissions = tracker.stop()

    print("Total energy consumed (kWh): ", emissions)


    return all_accuracies, all_f1, all_recall, all_precision, all_specificity, all_class_accuracies

# Main Code

if __name__ == "__main__":
    
   
    data = pd.read_csv('OriginalNetoDataset.csv')
    data = data.dropna()

    X_reference = data.iloc[:, :12].apply(pd.to_numeric, errors='coerce').values
    X_spectra = data.iloc[:, 12:-1].apply(pd.to_numeric, errors='coerce').values
    y = data.iloc[:, -1].astype('category').cat.codes.values

    if np.isnan(X_reference).any() or np.isnan(X_spectra).any():
        raise ValueError("Dataset contains NaN values.")

    scaler_ref = StandardScaler()
    X_reference_scaled = scaler_ref.fit_transform(X_reference)

    # You may also scale spectra if needed
    scaler_spec = StandardScaler()
    X_spectra_scaled = scaler_spec.fit_transform(X_spectra)

        
    
    X_train_ref, X_temp_ref, \
    X_train_spec, X_temp_spec, \
    y_train, y_temp = train_test_split(
        X_reference_scaled,
        X_spectra_scaled,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_val_ref, X_test_ref, \
    X_val_spec, X_test_spec, \
    y_val, y_test = train_test_split(
        X_temp_ref,
        X_temp_spec,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )


    X_combined = np.concatenate((X_train_ref, X_train_spec), axis=1)

    print("Before SMOTE class distribution:")
    print(pd.Series(y).value_counts())

    sm = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)

    X_balanced, y_balanced = sm.fit_resample(X_combined, y_train)
    
    print("After balanced: reference")
    print(X_balanced.shape)
    print("After balanced: spectra")
    print(y_balanced.shape)

    print("After SMOTE class distribution:")
    print(pd.Series(y_balanced).value_counts())

    # Separate back
    X_reference_bal = X_balanced[:, :12]
    X_spectra_bal = X_balanced[:, 12:-1]

    
    print("After balanced: reference")
    print(X_reference_bal)
    print("After balanced: spectra")
    print(X_spectra_bal)

    
    

    for i in range(len(X_spectra_bal)):
        X_spectra_bal[i] = add_gaussian_noise(
            X_spectra_bal[i],
            noise_std=0.01,
            prob=0.7
        )

        X_spectra_bal[i] = random_shift_broadening(
            X_spectra_bal[i],
            max_shift=3,
            prob=0.5
        )


    X_train_ref_tensor = torch.tensor(X_reference_bal, dtype=torch.float32).unsqueeze(1)
    X_train_spec_tensor = torch.tensor(X_spectra_bal, dtype=torch.float32).unsqueeze(1)

    X_val_ref_tensor = torch.tensor(X_val_ref, dtype=torch.float32).unsqueeze(1)
    X_val_spec_tensor = torch.tensor(X_val_spec, dtype=torch.float32).unsqueeze(1)

    X_test_ref_tensor = torch.tensor(X_test_ref, dtype=torch.float32).unsqueeze(1)
    X_test_spec_tensor = torch.tensor(X_test_spec, dtype=torch.float32).unsqueeze(1)

    y_train_tensor = torch.tensor(y_balanced, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Get class names in the same order as encoded labels
    class_names = sorted(data['Ingredient'].unique())
    print("Class mapping:")
    for i, name in enumerate(class_names):
        print(f"{i} -> {name}")
    
        
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_ref_tensor, X_train_spec_tensor,y_train_tensor)
    val_dataset = TensorDataset(X_val_ref_tensor, X_val_spec_tensor,y_val_tensor)
    test_dataset = TensorDataset(X_test_ref_tensor, X_test_spec_tensor,y_test_tensor)
    
    g = torch.Generator()
    g.manual_seed(42)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=g)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize model, criterion, and optimizer
    model = SpectralCNNLSTM(input_channels=1, num_filters=[16,64,256], kernel_sizes=[8, 4, 4],stride = [2,2,2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
        # ----------------------------
    # Start Energy Tracker
    # ----------------------------
    tracker = EmissionsTracker(
        project_name="DL_Model_Energy",
        output_dir="energy_logs"
    )

    tracker.start()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epoch=100)
    


