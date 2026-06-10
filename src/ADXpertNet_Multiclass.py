import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

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

    torch.use_deterministic_algorithms(True)

set_seed(42)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=518):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, dim)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# Multi-head Attention Block
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(x + attn_output)

# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, kv):
        attn_output, _ = self.cross_attn(q, kv, kv)
        return self.norm(q + attn_output)


class SpectralCNN(nn.Module):
    def __init__(self, input_channels=1, dropout_rate=0.3):
        super(SpectralCNN, self).__init__()

        # ---- Block 1 (Local features) ----
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool1d(2)

        # ---- Block 2 (Medium receptive field) ----
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool1d(2)

        # ---- Block 3 (Large receptive field) ----
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.dropout(x)

        return x
        



    
# Full Multimodal Model
class MultimodalFusionModel(nn.Module):
    def __init__(self, component_dim, spectral_dim, num_classes):
        super(MultimodalFusionModel, self).__init__()
        component_dim = 64
        spectral_dim = 64

        # --- Component branch ---
        self.component_proj = nn.Linear(12, component_dim)
        self.component_pos_enc = PositionalEncoding(component_dim, max_len=1)
        self.component_attn = MultiHeadSelfAttention(component_dim)

        self.conv_layers = SpectralCNN(input_channels=1)

        self.spectral_proj = nn.Linear(256, spectral_dim)
        
        self.spectral_pos_enc = PositionalEncoding(spectral_dim, max_len=64)
        self.spectral_attn = MultiHeadSelfAttention(spectral_dim)

        # --- Cross-Attention Fusion ---
        self.cross_attn = CrossAttention(spectral_dim)
        self.spectral_dim = spectral_dim  # <-- ADD THIS LINE



        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self.spectral_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, component_input, spectral_input):
    # --- Component branch ---
        comp = self.component_proj(component_input)  # (B, 1, 64)
        comp = self.component_pos_enc(comp)  # (B, 1, 64)
        comp = self.component_attn(comp)     # (B, 1, 64)

        x = spectral_input

        x = self.conv_layers(x)

        x = x.permute(0, 2, 1)

        x = self.spectral_proj(x)

        x = self.spectral_pos_enc(x)    # Add positional encoding

        x = self.spectral_attn(x)       # Self-attention: (B, L', 64)

        # --- Fusion via cross attention ---
        fused = self.cross_attn(x, comp)  # (B, L', 64)

        # --- Classification ---
        fused_flat = fused.reshape(fused.shape[0], -1)  # (B, L'*64)
        out = self.classifier(fused_flat)               # (B, 6)
        return out

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
                outputs = model(reference, spectra)
                loss = criterion(outputs, labels)

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
                    outputs = model(reference, spectra)
                    loss = criterion(outputs, labels)
                                    

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
                    
                    outputs = model(reference, spectra)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                                        
                    correct_test += (predicted == labels).sum().item()
                    total_test += labels.size(0)
                    
                    all_targets.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    

                    y_true = np.array(all_targets)
                    y_pred = np.array(all_preds)
                  
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
    

    return all_accuracies, all_f1, all_recall, all_precision, all_specificity, all_class_accuracies

# Main Code

if __name__ == "__main__":
    
   
    data = pd.read_csv('MilkSpectraDataset.csv')
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
    model = MultimodalFusionModel(component_dim=12, spectral_dim=518, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epoch=100)
    


