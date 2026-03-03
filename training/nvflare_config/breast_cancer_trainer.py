"""
NVFLARE Trainer para Wisconsin Breast Cancer Dataset
Implementa Client API con métricas médicas detalladas
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey
import os

class BreastCancerTrainer(Learner):
    """
    Trainer NVFLARE para clasificación de cancer de mama
    Cada instancia representa un hospital con datos locales
    """
    
    def __init__(
        self,
        model,
        client_id: int = 0,
        lr: float = 0.001,
        epochs: int = 3,
        batch_size: int = 32,
        dataset_path: str = "datasets/cancer"
    ):
        super().__init__()
        self.model = model
        self.client_id = client_id
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Cargar datos locales del hospital
        self.train_loader = self._load_local_data()
        
        print(f"\n[Hospital {self.client_id}] Trainer inicializado")
        print(f"  Device: {self.device}")
        print(f"  Datos locales: {len(self.train_loader.dataset)} muestras")
    
    def _load_local_data(self):
        """Cargar datos privados del hospital"""
        X_path = os.path.join(self.dataset_path, f"client_{self.client_id}_X.npy")
        y_path = os.path.join(self.dataset_path, f"client_{self.client_id}_y.npy")
        
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        
        # Convertir a tensors
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        # Crear dataset y dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return loader
    
    def train(self, model_learnable: ModelLearnable, abort_signal):
        """
        Entrenamiento local en el hospital
        
        Args:
            model_learnable: Modelo global recibido del servidor
            abort_signal: Signal para abortar entrenamiento
            
        Returns:
            ModelLearnable con pesos actualizados
        """
        print(f"\n[Hospital {self.client_id}] Iniciando entrenamiento local...")
        
        # Cargar pesos del modelo global
        if model_learnable:
            self._load_model_weights(model_learnable)
        
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if abort_signal.triggered:
                    return None
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Métricas
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += len(data)
            
            epoch_acc = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(self.train_loader)
            
            print(f"  Epoch {epoch+1}/{self.epochs}: "
                  f"Loss={avg_loss:.4f}, Acc={epoch_acc:.4f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Métricas finales
        final_acc = correct / total
        avg_loss = total_loss / (len(self.train_loader) * self.epochs)
        
        print(f"[Hospital {self.client_id}] Entrenamiento completado:")
        print(f"  Accuracy final: {final_acc:.4f}")
        print(f"  Loss promedio: {avg_loss:.4f}")
        
        # Retornar modelo actualizado
        return self._get_model_learnable()
    
    def validate(self, model_learnable: ModelLearnable, abort_signal):
        """
        Validación local con métricas médicas
        
        Returns:
            dict con métricas (accuracy, sensitivity, specificity)
        """
        if model_learnable:
            self._load_model_weights(model_learnable)
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calcular métricas médicas
        metrics = self._calculate_medical_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        print(f"\n[Hospital {self.client_id}] Validación local:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Sensitivity (detectar maligno): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (detectar benigno): {metrics['specificity']:.4f}")
        print(f"  TP={metrics['tp']}, TN={metrics['tn']}, "
              f"FP={metrics['fp']}, FN={metrics['fn']}")
        
        return metrics
    
    def _calculate_medical_metrics(self, predictions, targets):
        """
        Calcular métricas médicas específicas
        
        Clase 0 = Maligno (cáncer)
        Clase 1 = Benigno (no cáncer)
        """
        # Matriz de confusión
        tp = np.sum((predictions == 0) & (targets == 0))  # Maligno detectado
        tn = np.sum((predictions == 1) & (targets == 1))  # Benigno detectado
        fp = np.sum((predictions == 0) & (targets == 1))  # Falso positivo
        fn = np.sum((predictions == 1) & (targets == 0))  # Falso negativo (GRAVE)
        
        # Métricas
        accuracy = (tp + tn) / len(targets) if len(targets) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall para maligno
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall para benigno
        
        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def _load_model_weights(self, model_learnable: ModelLearnable):
        """Cargar pesos del modelo global"""
        weights = model_learnable.get(ModelLearnableKey.WEIGHTS)
        self.model.load_state_dict(weights)
    
    def _get_model_learnable(self):
        """Obtener modelo actualizado como ModelLearnable"""
        weights = self.model.state_dict()
        return ModelLearnable(weights={ModelLearnableKey.WEIGHTS: weights})

def create_trainer(model, client_id=0, **kwargs):
    """
    Factory function para crear trainer NVFLARE
    
    Args:
        model: Modelo PyTorch (BreastCancerMLP)
        client_id: ID del hospital (0, 1, 2)
        **kwargs: Parámetros adicionales (lr, epochs, batch_size)
    
    Returns:
        BreastCancerTrainer instance
    """
    return BreastCancerTrainer(
        model=model,
        client_id=client_id,
        lr=kwargs.get('lr', 0.001),
        epochs=kwargs.get('epochs', 3),
        batch_size=kwargs.get('batch_size', 32),
        dataset_path=kwargs.get('dataset_path', 'datasets/cancer')
    )
