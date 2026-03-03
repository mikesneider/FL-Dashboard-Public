"""
NVFLARE - Federated Learning con Imágenes de Ultrasonido
Breast Ultrasound Images Dataset (3 clases: benign, malignant, normal)

CARACTERÍSTICAS:
- Dataset: Imágenes de ultrasonido de mama de Kaggle
- Modelo: ResNet18 (Transfer Learning) con GPU
- 3 Hospitales con distribuciones NON-IID
- FL Training + Fine-Tuning
- Resultados guardados en: training_history_ultrasound_nvflare.json

ARQUITECTURA:
1. FL Training Phase: Hospitales entrenan colaborativamente → Modelo Global
2. Fine-tuning Phase: Cada hospital refina el modelo con datos adicionales
3. Resultados visibles en el Dashboard Flask

REQUISITOS:
- GPU con CUDA (recomendado)
- Dataset descargado (ejecutar: prepare_ultrasound_dataset.py)
- PyTorch, torchvision, Pillow
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from pathlib import Path

# Detectar raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
import os
import json
from datetime import datetime
import time
from pathlib import Path
from collections import Counter

# Verificar GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

def get_gpu_metrics():
    """Capturar métricas de GPU si está disponible"""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'device_name': 'CPU',
            'memory_allocated_mb': 0,
            'memory_reserved_mb': 0,
            'memory_free_mb': 0,
            'utilization_percent': 0
        }
    
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        memory_free = memory_total - memory_reserved
        
        # Utilización aproximada basada en memoria
        utilization = (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
        
        return {
            'available': True,
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated_mb': round(memory_allocated, 2),
            'memory_reserved_mb': round(memory_reserved, 2),
            'memory_total_mb': round(memory_total, 2),
            'memory_free_mb': round(memory_free, 2),
            'utilization_percent': round(utilization, 2)
        }
    except Exception as e:
        return {
            'available': False,
            'device_name': 'CPU',
            'error': str(e)
        }

class UltrasoundClient:
    """Cliente (Hospital) para Federated Learning con imágenes"""
    
    def __init__(self, client_id, train_dataset, finetune_dataset, num_classes=3):
        self.client_id = client_id
        self.device = DEVICE
        self.num_classes = num_classes
        
        # Crear DataLoaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=16,  # Ajustar según memoria GPU
            shuffle=True,
            num_workers=2,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
        
        self.finetune_loader = DataLoader(
            finetune_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )
        
        # Crear modelo
        self.model = self._create_model()
        
        # Imprimir estadísticas del cliente
        print(f"\n🏥 Hospital {client_id}:")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Fine-tuning samples: {len(finetune_dataset)}")
        
        # Mostrar distribución de clases
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        train_dist = Counter(train_labels)
        print(f"   Training distribution: {dict(train_dist)}")
    
    def _create_model(self):
        """Crea ResNet18 pre-entrenado con Transfer Learning"""
        # Cargar ResNet18 pre-entrenado
        model = models.resnet18(pretrained=True)
        
        # Modificar última capa para 3 clases
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.num_classes)
        )
        
        return model.to(self.device)
    
    def train_fl(self, epochs=3, lr=0.001):
        """Entrenamiento FL con datos de training"""
        start_time = time.time()
        
        # Capturar métricas iniciales de GPU
        gpu_metrics_start = get_gpu_metrics()
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        # Capturar métricas finales de GPU
        gpu_metrics_end = get_gpu_metrics()
        
        training_time = time.time() - start_time
        return training_time, gpu_metrics_start, gpu_metrics_end
    
    def finetune(self, epochs=2, lr=0.0001):
        """Fine-tuning con datos adicionales"""
        start_time = time.time()
        
        # Capturar métricas iniciales de GPU
        gpu_metrics_start = get_gpu_metrics()
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in self.finetune_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(self.finetune_loader)
            accuracy = correct / total
            
            print(f"  Fine-tune Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        # Capturar métricas finales de GPU
        gpu_metrics_end = get_gpu_metrics()
        
        finetune_time = time.time() - start_time
        return finetune_time, gpu_metrics_start, gpu_metrics_end
    
    def evaluate(self, test_loader):
        """Evaluar modelo en test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calcular métricas (multiclass)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Calcular métricas por clase
        class_metrics = {}
        for class_idx in range(self.num_classes):
            tp = ((all_preds == class_idx) & (all_labels == class_idx)).sum()
            tn = ((all_preds != class_idx) & (all_labels != class_idx)).sum()
            fp = ((all_preds == class_idx) & (all_labels != class_idx)).sum()
            fn = ((all_preds != class_idx) & (all_labels == class_idx)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f'class_{class_idx}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        return {
            'accuracy': float(accuracy),
            'class_metrics': class_metrics
        }
    
    def get_weights(self):
        """Obtener pesos del modelo"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """Establecer pesos del modelo"""
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})


class UltrasoundServer:
    """Servidor central para agregación FedAvg"""
    
    def __init__(self, num_classes=3):
        self.device = DEVICE
        self.num_classes = num_classes
        self.model = self._create_model()
        self.history = []
    
    def _create_model(self):
        """Crea modelo global"""
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.num_classes)
        )
        return model.to(self.device)
    
    def aggregate(self, client_weights_list):
        """Agregación FedAvg: Promedio ponderado de pesos"""
        global_weights = {}
        num_clients = len(client_weights_list)
        
        # Inicializar con el primer cliente
        for key in client_weights_list[0].keys():
            global_weights[key] = torch.zeros_like(client_weights_list[0][key], dtype=torch.float32)
        
        # Sumar pesos de todos los clientes
        for client_weights in client_weights_list:
            for key in global_weights.keys():
                # Convertir a float32 para la agregación
                global_weights[key] += client_weights[key].float() / num_clients
        
        # Convertir de vuelta al tipo original
        for key in global_weights.keys():
            original_dtype = client_weights_list[0][key].dtype
            if original_dtype in [torch.int32, torch.int64, torch.long]:
                # Para tipos enteros (como num_batches_tracked), usar el máximo
                global_weights[key] = torch.max(
                    torch.stack([w[key] for w in client_weights_list])
                ).to(original_dtype)
            else:
                # Para tipos flotantes, mantener el promedio
                global_weights[key] = global_weights[key].to(original_dtype)
        
        return global_weights
    
    def get_weights(self):
        """Obtener pesos del modelo global"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """Establecer pesos del modelo global"""
        self.model.load_state_dict({k: v.to(self.device) for k, v in weights.items()})
    
    def evaluate(self, test_loader):
        """Evaluar modelo global en test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Calcular métricas por clase
        class_metrics = {}
        for class_idx in range(self.num_classes):
            tp = ((all_preds == class_idx) & (all_labels == class_idx)).sum()
            tn = ((all_preds != class_idx) & (all_labels != class_idx)).sum()
            fp = ((all_preds == class_idx) & (all_labels != class_idx)).sum()
            fn = ((all_preds != class_idx) & (all_labels == class_idx)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f'class_{class_idx}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        return {
            'accuracy': float(accuracy),
            'class_metrics': class_metrics
        }


def load_ultrasound_dataset():
    """Carga el dataset de ultrasonido y lo prepara para FL"""
    print("\n📥 Cargando dataset de imágenes de ultrasonido...")
    
    # Buscar el dataset
    data_dir = Path("datasets/breast_ultrasound")
    
    # Buscar el directorio Dataset
    dataset_path = None
    for root, dirs, files in os.walk(data_dir):
        if 'benign' in dirs and 'malignant' in dirs and 'normal' in dirs:
            dataset_path = Path(root)
            break
    
    if not dataset_path:
        raise FileNotFoundError(
            "❌ Dataset no encontrado. Ejecuta primero:\n"
            "   python prepare_ultrasound_dataset.py"
        )
    
    print(f"✅ Dataset encontrado: {dataset_path}")
    
    # Transformaciones para imágenes
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar dataset completo
    full_dataset = ImageFolder(root=str(dataset_path), transform=transform)
    
    print(f"✅ Total de imágenes: {len(full_dataset)}")
    print(f"   Clases: {full_dataset.classes}")
    print(f"   Class to idx: {full_dataset.class_to_idx}")
    
    return full_dataset


def split_dataset_non_iid(full_dataset, num_clients=3):
    """Divide el dataset en splits NON-IID para cada hospital"""
    print(f"\n🔀 Dividiendo dataset en {num_clients} hospitales (NON-IID)...")
    
    # Obtener índices por clase
    targets = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
    class_indices = {c: np.where(targets == c)[0] for c in range(len(full_dataset.classes))}
    
    print(f"   Distribución original por clase:")
    for class_idx, indices in class_indices.items():
        print(f"     Clase {class_idx} ({full_dataset.classes[class_idx]}): {len(indices)} imágenes")
    
    # Configuración NON-IID
    # Hospital 0: 50% benign (0), 25% malignant (1), 25% normal (2)
    # Hospital 1: 25% benign, 50% malignant, 25% normal
    # Hospital 2: 33% benign, 34% malignant, 33% normal
    
    distributions = [
        [0.50, 0.25, 0.25],  # Hospital 0 (screening)
        [0.25, 0.50, 0.25],  # Hospital 1 (oncology)
        [0.33, 0.34, 0.33]   # Hospital 2 (general)
    ]
    
    # Crear splits por cliente
    client_train_indices = [[] for _ in range(num_clients)]
    client_finetune_indices = [[] for _ in range(num_clients)]
    test_indices = []
    
    for class_idx, indices in class_indices.items():
        np.random.shuffle(indices)
        
        # Reservar 20% para test global
        test_size = int(len(indices) * 0.20)
        test_indices.extend(indices[:test_size])
        remaining = indices[test_size:]
        
        # Dividir el 80% restante entre clientes según distribución
        total_per_client = [int(len(remaining) * dist[class_idx]) for dist in distributions]
        
        # Ajustar para que sume exactamente len(remaining)
        diff = len(remaining) - sum(total_per_client)
        total_per_client[0] += diff
        
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + total_per_client[client_id]
            client_indices = remaining[start_idx:end_idx]
            
            # Split 85% training, 15% fine-tuning
            split_point = int(len(client_indices) * 0.85)
            client_train_indices[client_id].extend(client_indices[:split_point])
            client_finetune_indices[client_id].extend(client_indices[split_point:])
            
            start_idx = end_idx
    
    # Crear Subsets
    print(f"\n📊 Splits creados:")
    client_train_datasets = []
    client_finetune_datasets = []
    
    for client_id in range(num_clients):
        train_dataset = Subset(full_dataset, client_train_indices[client_id])
        finetune_dataset = Subset(full_dataset, client_finetune_indices[client_id])
        
        client_train_datasets.append(train_dataset)
        client_finetune_datasets.append(finetune_dataset)
        
        print(f"   Hospital {client_id}: {len(train_dataset)} train, {len(finetune_dataset)} finetune")
    
    test_dataset = Subset(full_dataset, test_indices)
    print(f"   Test global: {len(test_dataset)} imágenes")
    
    # Crear test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return client_train_datasets, client_finetune_datasets, test_loader, full_dataset.classes


def run_federated_learning(num_rounds=5, num_clients=3):
    """Ejecuta el proceso completo de FL + Fine-tuning"""
    
    print("=" * 80)
    print("🚀 FEDERATED LEARNING - BREAST ULTRASOUND IMAGES")
    print("=" * 80)
    
    # Cargar y dividir dataset
    full_dataset = load_ultrasound_dataset()
    client_train_datasets, client_finetune_datasets, test_loader, class_names = split_dataset_non_iid(
        full_dataset, num_clients
    )
    
    # Crear servidor
    server = UltrasoundServer(num_classes=len(class_names))
    
    # Crear clientes
    clients = []
    for i in range(num_clients):
        client = UltrasoundClient(
            client_id=i,
            train_dataset=client_train_datasets[i],
            finetune_dataset=client_finetune_datasets[i],
            num_classes=len(class_names)
        )
        clients.append(client)
    
    # Inicializar JSON de historial
    history = []
    
    # Round 0: Evaluación inicial
    print("\n" + "=" * 80)
    print("📊 ROUND 0: Evaluación Inicial (modelo pre-entrenado)")
    print("=" * 80)
    
    initial_metrics = server.evaluate(test_loader)
    print(f"Modelo Global Inicial - Accuracy: {initial_metrics['accuracy']:.4f}")
    
    history.append({
        'round': 0,
        'phase': 'fl_training',
        'global_metrics': initial_metrics,
        'client_metrics': [],
        'timing': {'round_time': 0}
    })
    
    # ========== FASE 1: FL TRAINING ==========
    fl_training_start = time.time()
    
    for round_num in range(1, num_rounds + 1):
        print("\n" + "=" * 80)
        print(f"📡 ROUND {round_num}/{num_rounds}: FL Training")
        print("=" * 80)
        
        round_start = time.time()
        
        # Distribuir modelo global a clientes
        global_weights = server.get_weights()
        for client in clients:
            client.set_weights(global_weights)
        
        # Entrenar cada cliente
        client_weights_list = []
        client_metrics_list = []
        
        for client in clients:
            print(f"\n🏥 Hospital {client.client_id} entrenando...")
            train_time, gpu_start, gpu_end = client.train_fl(epochs=2, lr=0.001)
            
            # Evaluar cliente
            client_metrics = client.evaluate(test_loader)
            client_metrics['client_id'] = client.client_id
            client_metrics['train_time'] = train_time
            client_metrics['gpu_metrics'] = {
                'start': gpu_start,
                'end': gpu_end,
                'peak_memory_mb': gpu_end.get('memory_reserved_mb', 0)
            }
            
            print(f"   Accuracy: {client_metrics['accuracy']:.4f}")
            print(f"   Training time: {train_time:.2f}s")
            if gpu_end['available']:
                print(f"   GPU Memory: {gpu_end['memory_reserved_mb']:.2f} MB")
            
            client_weights_list.append(client.get_weights())
            client_metrics_list.append(client_metrics)
        
        # Agregación FedAvg
        print("\n🔄 Agregando modelos (FedAvg)...")
        aggregated_weights = server.aggregate(client_weights_list)
        server.set_weights(aggregated_weights)
        
        # Evaluar modelo global
        global_metrics = server.evaluate(test_loader)
        print(f"\n✅ Modelo Global Round {round_num} - Accuracy: {global_metrics['accuracy']:.4f}")
        
        round_time = time.time() - round_start
        
        # Guardar en historial
        history.append({
            'round': round_num,
            'phase': 'fl_training',
            'global_metrics': global_metrics,
            'client_metrics': client_metrics_list,
            'timing': {
                'round_time': round_time,
                'total_fl_time': time.time() - fl_training_start
            }
        })
    
    fl_training_time = time.time() - fl_training_start
    
    # ========== FASE 2: FINE-TUNING ==========
    print("\n" + "=" * 80)
    print(f"🎯 FINE-TUNING: Refinamiento local por hospital")
    print("=" * 80)
    
    finetune_start = time.time()
    
    # Cada cliente refina con sus datos de fine-tuning
    refined_metrics_list = []
    
    for client in clients:
        print(f"\n🏥 Hospital {client.client_id} fine-tuning...")
        
        # Copiar modelo global antes de fine-tuning
        client.set_weights(server.get_weights())
        
        # Fine-tuning
        ft_time, gpu_start, gpu_end = client.finetune(epochs=2, lr=0.0001)
        
        # Evaluar modelo refinado
        refined_metrics = client.evaluate(test_loader)
        refined_metrics['client_id'] = client.client_id
        refined_metrics['finetune_time'] = ft_time
        refined_metrics['gpu_metrics'] = {
            'start': gpu_start,
            'end': gpu_end,
            'peak_memory_mb': gpu_end.get('memory_reserved_mb', 0)
        }
        
        print(f"   Refined Accuracy: {refined_metrics['accuracy']:.4f}")
        print(f"   Fine-tuning time: {ft_time:.2f}s")
        if gpu_end['available']:
            print(f"   GPU Memory: {gpu_end['memory_reserved_mb']:.2f} MB")
        
        refined_metrics_list.append(refined_metrics)
    
    finetune_total_time = time.time() - finetune_start
    
    # Guardar fase de fine-tuning
    history.append({
        'round': num_rounds + 1,
        'phase': 'fine_tuning',
        'refined_models': refined_metrics_list,
        'timing': {
            'finetune_total_time': finetune_total_time,
            'hospitals': {
                f'hospital_{m["client_id"]}': m['finetune_time'] 
                for m in refined_metrics_list
            }
        }
    })
    
    # ========== GUARDAR RESULTADOS ==========
    output_file = PROJECT_ROOT / 'training_history_ultrasound_nvflare.json'
    
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"\n📊 Resultados guardados en: {output_file}")
    print(f"\n⏱️ Tiempos de ejecución:")
    print(f"   FL Training: {fl_training_time:.2f}s")
    print(f"   Fine-tuning: {finetune_total_time:.2f}s")
    print(f"   Total: {fl_training_time + finetune_total_time:.2f}s")
    
    print(f"\n📈 Métricas finales:")
    print(f"   Modelo Global: Accuracy = {global_metrics['accuracy']:.4f}")
    print(f"   Modelos Refinados:")
    for m in refined_metrics_list:
        print(f"     Hospital {m['client_id']}: Accuracy = {m['accuracy']:.4f}")
    
    print(f"\n🎯 Clases del dataset: {class_names}")
    print(f"   0: {class_names[0]}")
    print(f"   1: {class_names[1]}")
    print(f"   2: {class_names[2]}")
    
    print(f"\n💡 Ver resultados en el Dashboard:")
    print(f"   python dashboard_flask_cancer.py")
    print(f"   http://127.0.0.1:5000")
    print(f"   Selecciona: 'Breast Ultrasound (Images)' en el menú")


if __name__ == "__main__":
    # Ejecutar FL con 5 rounds y 3 hospitales
    run_federated_learning(num_rounds=5, num_clients=3)
