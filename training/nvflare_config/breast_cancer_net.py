"""
Red neuronal MLP para Wisconsin Breast Cancer Dataset
Compatible con NVFLARE Client API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BreastCancerMLP(nn.Module):
    """
    MLP para clasificación binaria de cancer de mama
    Input: 30 features → Hidden: 64 → 32 → Output: 2 classes
    """
    def __init__(self, input_size=30, hidden1=64, hidden2=32, num_classes=2):
        super(BreastCancerMLP, self).__init__()
        
        # Capas fully connected
        self.fc1 = nn.Linear(input_size, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        # Inicialización He
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: tensor (batch_size, 30)
        Returns:
            tensor (batch_size, 2) con logits
        """
        # Capa 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Capa 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Capa 3 (output)
        x = self.fc3(x)
        
        return x

def create_model():
    """Factory function para NVFLARE"""
    return BreastCancerMLP(input_size=30, hidden1=64, hidden2=32, num_classes=2)
