"""
NVFLARE - Federated Learning + Fine-tuning
Breast Cancer Detection con medición de tiempos

ARQUITECTURA:
1. FL Training Phase: Hospitales entrenan colaborativamente → Modelo Global
2. Fine-tuning Phase: Cada hospital refina el modelo con datos adicionales
3. Comparación: Modelo Global vs Modelos Refinados

MÉTRICAS REGISTRADAS:
- Tiempos de entrenamiento por fase y por hospital
- Accuracy, Sensitivity, Specificity
- Matriz de confusión (TP, TN, FP, FN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from datetime import datetime
import time
from pathlib import Path

# Detectar raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Importar arquitectura del modelo
import sys
sys.path.append(str(PROJECT_ROOT / 'nvflare_config'))
from breast_cancer_net import create_model

class BreastCancerClient:
    """Cliente (Hospital) con capacidades de FL Training y Fine-tuning"""
    
    def __init__(self, client_id, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.model = create_model().to(device)
        
        # Cargar datos de FL training
        self.X_train = np.load(f'datasets/cancer/client_{client_id}_X.npy')
        self.y_train = np.load(f'datasets/cancer/client_{client_id}_y.npy')
        
        # Cargar datos de fine-tuning
        self.X_finetune = np.load(f'datasets/cancer/client_{client_id}_finetune_X.npy')
        self.y_finetune = np.load(f'datasets/cancer/client_{client_id}_finetune_y.npy')
        
        self.train_loader = self._create_dataloader(self.X_train, self.y_train)
        self.finetune_loader = self._create_dataloader(self.X_finetune, self.y_finetune, batch_size=8)
    
    def _create_dataloader(self, X, y, batch_size=16, shuffle=True):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_fl(self, epochs=3, lr=0.001):
        """Entrenamiento FL con datos de training"""
        start_time = time.time()
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        training_time = time.time() - start_time
        return training_time
    
    def finetune(self, epochs=2, lr=0.0001):
        """Fine-tuning con datos adicionales"""
        start_time = time.time()
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in self.finetune_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            avg_loss = total_loss / len(self.finetune_loader)
            accuracy = correct / total
            
            print(f"  FT Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        finetune_time = time.time() - start_time
        return finetune_time
    
    def get_weights(self):
        """Obtener pesos del modelo"""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """Establecer pesos del modelo"""
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                if v.shape == ():  # Scalar
                    state_dict[k] = torch.tensor(v.item())
                else:
                    state_dict[k] = torch.FloatTensor(v)
            else:
                state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)
    
    def evaluate_local(self):
        """Evaluar modelo en datos locales de training"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        return self._calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcular métricas médicas"""
        tp = np.sum((y_pred == 0) & (y_true == 0))  # Maligno detectado
        tn = np.sum((y_pred == 1) & (y_true == 1))  # Benigno detectado
        fp = np.sum((y_pred == 0) & (y_true == 1))  # Falsa alarma
        fn = np.sum((y_pred == 1) & (y_true == 0))  # Cáncer NO detectado
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }


class BreastCancerServer:
    """Servidor NVFLARE para coordinar FL y Fine-tuning"""
    
    def __init__(self, n_clients=3, device='cpu'):
        self.n_clients = n_clients
        self.device = device
        self.global_model = create_model().to(device)
        
        # Cargar datos de test
        self.X_test = np.load('datasets/cancer/test_X.npy')
        self.y_test = np.load('datasets/cancer/test_y.npy')
        self.test_loader = self._create_dataloader(self.X_test, self.y_test)
    
    def _create_dataloader(self, X, y, batch_size=32):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def federated_averaging(self, client_weights_list):
        """Promedio federado de pesos (FedAvg)"""
        avg_weights = {}
        
        for key in client_weights_list[0].keys():
            avg_weights[key] = np.mean(
                [client_weights[key] for client_weights in client_weights_list],
                axis=0
            )
        
        return avg_weights
    
    def get_global_weights(self):
        """Obtener pesos del modelo global"""
        return {k: v.cpu().numpy() for k, v in self.global_model.state_dict().items()}
    
    def set_global_weights(self, weights):
        """Establecer pesos del modelo global"""
        state_dict = {}
        for k, v in weights.items():
            if isinstance(v, np.ndarray):
                if v.shape == ():  # Scalar
                    state_dict[k] = torch.tensor(v.item())
                else:
                    state_dict[k] = torch.FloatTensor(v)
            else:
                state_dict[k] = torch.tensor(v)
        self.global_model.load_state_dict(state_dict)
    
    def evaluate_global(self):
        """Evaluar modelo global en test set"""
        self.global_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.global_model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        
        # Calcular métricas
        tp = np.sum((y_pred == 0) & (y_true == 0))
        tn = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 0) & (y_true == 1))
        fn = np.sum((y_pred == 1) & (y_true == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def run_federated_learning(self, n_rounds=5, fl_epochs=3, ft_epochs=2):
        """
        Ejecutar FL completo + Fine-tuning
        
        FASE 1: FL Training (n_rounds)
        FASE 2: Fine-tuning por hospital
        """
        print("\n" + "="*70)
        print("  NVFLARE - FEDERATED LEARNING + FINE-TUNING")
        print("="*70 + "\n")
        
        print(f"[Setup] {self.n_clients} hospitales inicializados\n")
        
        # Crear clientes
        clients = [BreastCancerClient(i, self.device) for i in range(self.n_clients)]
        
        history = []
        total_start_time = time.time()
        
        # ====================================================================
        # FASE 1: FEDERATED LEARNING
        # ====================================================================
        print("="*70)
        print("  FASE 1: FEDERATED LEARNING")
        print("="*70 + "\n")
        
        # Evaluación inicial
        print("[Round 0] Evaluación inicial (modelo sin entrenar)...")
        global_metrics = self.evaluate_global()
        print(f"  Modelo Global:")
        print(f"    Accuracy: {global_metrics['accuracy']:.4f}")
        print(f"    Sensitivity: {global_metrics['sensitivity']:.4f}")
        print(f"    Specificity: {global_metrics['specificity']:.4f}\n")
        
        history.append({
            'round': 0,
            'phase': 'fl_training',
            'global_metrics': global_metrics,
            'client_metrics': [],
            'timing': {'fl_training_time': 0.0}
        })
        
        # Rounds de FL
        for round_num in range(1, n_rounds + 1):
            round_start_time = time.time()
            
            print("="*70)
            print(f"  ROUND {round_num}/{n_rounds} - FL TRAINING")
            print("="*70 + "\n")
            
            # Distribuir modelo global a clientes
            global_weights = self.get_global_weights()
            for client in clients:
                client.set_weights(global_weights)
            
            # Entrenamiento local en cada hospital
            client_weights_list = []
            client_metrics_list = []
            hospital_times = {}
            
            for client in clients:
                print(f"[Hospital {client.client_id}] Entrenamiento FL local...")
                fl_time = client.train_fl(epochs=fl_epochs)
                hospital_times[f'hospital_{client.client_id}'] = fl_time
                
                # Evaluar localmente
                local_metrics = client.evaluate_local()
                print(f"[Hospital {client.client_id}] Métricas locales:")
                print(f"  Accuracy: {local_metrics['accuracy']:.4f}, " 
                      f"Sensitivity: {local_metrics['sensitivity']:.4f}, "
                      f"Specificity: {local_metrics['specificity']:.4f}\n")
                
                client_weights_list.append(client.get_weights())
                client_metrics_list.append({
                    'client_id': client.client_id,
                    'metrics': local_metrics,
                    'fl_training_time': fl_time
                })
            
            # Agregación federada
            print("[Aggregation] Promediando modelos de hospitales...")
            aggregated_weights = self.federated_averaging(client_weights_list)
            self.set_global_weights(aggregated_weights)
            print("  [OK] Modelo global actualizado\n")
            
            # Evaluación global
            print(f"[Evaluation] Modelo Global - Round {round_num}...")
            global_metrics = self.evaluate_global()
            print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
            print(f"  Sensitivity: {global_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {global_metrics['specificity']:.4f}")
            print(f"  TP: {global_metrics['tp']}, TN: {global_metrics['tn']}, "
                  f"FP: {global_metrics['fp']}, FN: {global_metrics['fn']}\n")
            
            round_time = time.time() - round_start_time
            
            history.append({
                'round': round_num,
                'phase': 'fl_training',
                'global_metrics': global_metrics,
                'client_metrics': client_metrics_list,
                'timing': {
                    'round_time': round_time,
                    'hospitals': hospital_times
                }
            })
        
        fl_total_time = time.time() - total_start_time
        
        # ====================================================================
        # FASE 2: FINE-TUNING
        # ====================================================================
        print("\n" + "="*70)
        print("  FASE 2: FINE-TUNING (Refinar modelo con datos adicionales)")
        print("="*70 + "\n")
        
        ft_start_time = time.time()
        
        # Cada hospital refina el modelo global con sus datos de fine-tuning
        refined_models_metrics = []
        ft_hospital_times = {}
        
        for client in clients:
            print(f"[Hospital {client.client_id}] Fine-tuning con datos adicionales...")
            
            # Cargar modelo global como punto de partida
            client.set_weights(self.get_global_weights())
            
            # Fine-tuning
            ft_time = client.finetune(epochs=ft_epochs, lr=0.0001)
            ft_hospital_times[f'hospital_{client.client_id}'] = ft_time
            
            # Evaluar modelo refinado
            refined_metrics = client.evaluate_local()
            print(f"[Hospital {client.client_id}] Modelo refinado:")
            print(f"  Accuracy: {refined_metrics['accuracy']:.4f}, "
                  f"Sensitivity: {refined_metrics['sensitivity']:.4f}, "
                  f"Specificity: {refined_metrics['specificity']:.4f}\n")
            
            refined_models_metrics.append({
                'client_id': client.client_id,
                'refined_metrics': refined_metrics,
                'finetune_time': ft_time
            })
        
        ft_total_time = time.time() - ft_start_time
        total_time = time.time() - total_start_time
        
        # Guardar resultados de fine-tuning
        history.append({
            'round': n_rounds + 1,
            'phase': 'fine_tuning',
            'global_metrics': global_metrics,  # Modelo global sin fine-tuning
            'refined_models': refined_models_metrics,
            'timing': {
                'finetune_total_time': ft_total_time,
                'hospitals': ft_hospital_times
            }
        })
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        print("="*70)
        print("  ENTRENAMIENTO COMPLETADO")
        print("="*70 + "\n")
        
        print("📊 Resultados Finales:\n")
        print("  • Modelo Global (sin fine-tuning):")
        print(f"    - Accuracy: {global_metrics['accuracy']:.4f}")
        print(f"    - Sensitivity: {global_metrics['sensitivity']:.4f}")
        print(f"    - Specificity: {global_metrics['specificity']:.4f}\n")
        
        print("  • Modelos Refinados (con fine-tuning):")
        for refined in refined_models_metrics:
            m = refined['refined_metrics']
            print(f"    - Hospital {refined['client_id']}: Acc={m['accuracy']:.4f}, "
                  f"Sens={m['sensitivity']:.4f}, Spec={m['specificity']:.4f}")
        
        print(f"\n⏱️  Tiempos de Ejecución:")
        print(f"    - FL Training: {fl_total_time:.2f} segundos")
        print(f"    - Fine-tuning: {ft_total_time:.2f} segundos")
        print(f"    - Total: {total_time:.2f} segundos")
        
        print(f"\n📁 Historial guardado: training_history_cancer_nvflare.json")
        print(f"\n🏥 Privacidad: Datos nunca salieron de cada hospital\n")
        
        # Guardar historial con tiempos
        with open('training_history_cancer_nvflare.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history


if __name__ == "__main__":
    # Verificar datasets
    if not os.path.exists('datasets/cancer/client_0_finetune_X.npy'):
        print("\n❌ Error: Datasets de fine-tuning no encontrados")
        print("   Ejecuta: python download_dataset_cancer_v2.py\n")
        exit(1)
    
    # Ejecutar FL + Fine-tuning
    server = BreastCancerServer(n_clients=3)
    server.run_federated_learning(n_rounds=5, fl_epochs=3, ft_epochs=2)
