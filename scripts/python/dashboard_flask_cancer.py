"""
============================================================================
DASHBOARD FLASK - BREAST CANCER FEDERATED LEARNING
Dashboard interactivo con visualizaciones en tiempo real
============================================================================
"""

from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime

# Detectar raíz del proyecto (dos niveles arriba desde scripts/python/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"📁 Raíz del proyecto: {PROJECT_ROOT}")

# Configurar Flask con rutas a templates en la raíz del proyecto
app = Flask(__name__, 
            template_folder=str(PROJECT_ROOT / 'templates'),
            static_folder=str(PROJECT_ROOT / 'static') if (PROJECT_ROOT / 'static').exists() else None)

# Configuración de demos disponibles
DEMOS_CONFIG = {
    'cancer': {
        'name': 'Breast Cancer (Tabular)',
        'file': 'training_history_cancer_nvflare.json',
        'description': 'Clasificación binaria con datos tabulares (30 features)',
        'metrics_type': 'binary',
        'classes': ['Benign', 'Malignant']
    },
    'ultrasound': {
        'name': 'Breast Ultrasound (Images)',
        'file': 'training_history_ultrasound_nvflare.json',
        'description': 'Clasificación multiclase con imágenes de ultrasonido',
        'metrics_type': 'multiclass',
        'classes': ['Benign', 'Malignant', 'Normal']
    }
}

REFRESH_INTERVAL = 5000  # Actualizar cada 5 segundos

def load_training_history(demo_type='cancer', project_root=None):
    """Carga el historial de entrenamiento desde JSON según el tipo de demo
    
    Args:
        demo_type: Tipo de demo ('cancer' o 'ultrasound')
        project_root: Ruta raíz del proyecto (opcional). Si no se proporciona, usa PROJECT_ROOT del módulo
    """
    if demo_type not in DEMOS_CONFIG:
        print(f"⚠️ demo_type '{demo_type}' no está en DEMOS_CONFIG")
        return None
    
    history_file = DEMOS_CONFIG[demo_type]['file']
    
    # Usar project_root proporcionado o el del módulo
    root = Path(project_root) if project_root else PROJECT_ROOT
    
    # Construir ruta absoluta desde la raíz del proyecto
    history_path = root / history_file
    
    if not history_path.exists():
        print(f"⚠️ Archivo no encontrado: {history_path}")
        print(f"   Root usado: {root}")
        print(f"   Archivo buscado: {history_file}")
        return None
    
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ Archivo cargado exitosamente: {history_path} ({len(data)} entradas)")
            return data
    except Exception as e:
        print(f"❌ Error loading history for {demo_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_convergence_plot(history, metrics_type='binary'):
    """Gráfica de convergencia del modelo global"""
    if not history:
        return None
    
    # Filtrar solo entradas de FL training (que tienen global_metrics)
    fl_entries = [entry for entry in history if entry.get('phase') == 'fl_training' and 'global_metrics' in entry]
    
    if not fl_entries:
        print("⚠️ No se encontraron entradas de FL training con global_metrics")
        return None
    
    rounds = [entry['round'] for entry in fl_entries]
    accuracy = [entry['global_metrics']['accuracy'] * 100 for entry in fl_entries]
    
    # Para métricas binarias usamos sensitivity/specificity
    # Para multiclase usamos precision/recall promedio
    if metrics_type == 'binary':
        sensitivity = [entry['global_metrics']['sensitivity'] * 100 for entry in fl_entries]
        specificity = [entry['global_metrics']['specificity'] * 100 for entry in fl_entries]
    else:
        # Para multiclase, calcular precision/recall promedio de todas las clases
        sensitivity = []  # Será recall promedio
        specificity = []  # Será precision promedio
        for entry in fl_entries:
            if 'class_metrics' in entry['global_metrics']:
                class_metrics = entry['global_metrics']['class_metrics']
                avg_recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics)
                avg_precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics)
                sensitivity.append(avg_recall * 100)
                specificity.append(avg_precision * 100)
            else:
                sensitivity.append(0)
                specificity.append(0)
    
    fig = go.Figure()
    
    # Accuracy
    fig.add_trace(go.Scatter(
        x=rounds, y=accuracy,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Sensitivity/Recall
    metric1_name = 'Sensitivity' if metrics_type == 'binary' else 'Recall (Avg)'
    fig.add_trace(go.Scatter(
        x=rounds, y=sensitivity,
        mode='lines+markers',
        name=metric1_name,
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    
    # Specificity/Precision
    metric2_name = 'Specificity' if metrics_type == 'binary' else 'Precision (Avg)'
    fig.add_trace(go.Scatter(
        x=rounds, y=specificity,
        mode='lines+markers',
        name=metric2_name,
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Convergencia del Modelo Global',
        xaxis_title='Round',
        yaxis_title='Porcentaje (%)',
        yaxis=dict(range=[0, 105]),
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_comparison_plot(history):
    """Gráfica de comparación entre hospitales y modelo global"""
    if not history or len(history) == 0:
        return None
    
    # Buscar último entry de FL training (no fine-tuning)
    last_entry = None
    for entry in reversed(history):
        if entry.get('phase') == 'fl_training' and 'client_metrics' in entry and len(entry['client_metrics']) > 0:
            last_entry = entry
            break
    
    if not last_entry:
        return None
    
    # Datos
    hospitals = ['Hospital 0', 'Hospital 1', 'Hospital 2', 'Modelo Global']
    
    accuracy = []
    sensitivity = []
    specificity = []
    
    # Métricas de hospitales
    for client in last_entry['client_metrics']:
        # Manejar diferentes estructuras de JSON (cancer vs ultrasound)
        if 'metrics' in client:
            m = client['metrics']
            accuracy.append(m['accuracy'] * 100)
            sensitivity.append(m.get('sensitivity', 0) * 100)
            specificity.append(m.get('specificity', 0) * 100)
        else:
            accuracy.append(client.get('accuracy', 0) * 100)
            # Para ultrasound (multiclase), calculamos promedios de las clases
            if 'class_metrics' in client:
                classes = client['class_metrics']
                avg_recall = sum(c.get('recall', 0) for c in classes.values()) / len(classes) if classes else 0
                sensitivity.append(avg_recall * 100)
                # Specificity no es directo en multiclase, usamos 0 o calculamos aproximado
                specificity.append(0)
            else:
                sensitivity.append(0)
                specificity.append(0)
    
    # Métricas globales
    gm = last_entry['global_metrics']
    accuracy.append(gm['accuracy'] * 100)
    
    # Manejar diferentes estructuras para métricas globales
    if 'sensitivity' in gm:
        sensitivity.append(gm['sensitivity'] * 100)
        specificity.append(gm['specificity'] * 100)
    elif 'class_metrics' in gm:
        classes = gm['class_metrics']
        avg_recall = sum(c.get('recall', 0) for c in classes.values()) / len(classes) if classes else 0
        sensitivity.append(avg_recall * 100)
        specificity.append(0)
    else:
        sensitivity.append(0)
        specificity.append(0)
    
    fig = go.Figure()
    
    # Accuracy
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=hospitals,
        y=accuracy,
        marker_color='#1f77b4',
        text=[f'{v:.2f}%' for v in accuracy],
        textposition='outside'
    ))
    
    # Sensitivity
    fig.add_trace(go.Bar(
        name='Sensitivity',
        x=hospitals,
        y=sensitivity,
        marker_color='#ff7f0e',
        text=[f'{v:.2f}%' for v in sensitivity],
        textposition='outside'
    ))
    
    # Specificity
    fig.add_trace(go.Bar(
        name='Specificity',
        x=hospitals,
        y=specificity,
        marker_color='#2ca02c',
        text=[f'{v:.2f}%' for v in specificity],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Comparación Hospitales vs Global (Round {last_entry["round"]})',
        yaxis_title='Porcentaje (%)',
        yaxis=dict(range=[0, 110]),
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_confusion_matrix_plot(history):
    """Gráfica de matriz de confusión"""
    if not history or len(history) == 0:
        return None
    
    last_entry = history[-1]
    gm = last_entry['global_metrics']
    
    # Matriz de confusión
    z = [
        [gm['tp'], gm['fn']],  # Predicho Maligno
        [gm['fp'], gm['tn']]   # Predicho Benigno
    ]
    
    # Etiquetas
    x_labels = ['Real: Maligno', 'Real: Benigno']
    y_labels = ['Pred: Maligno', 'Pred: Benigno']
    
    # Anotaciones
    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            color = 'white' if val > 30 else 'black'
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f'<b>{val}</b>',
                    font=dict(size=24, color=color),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Casos")
    ))
    
    fig.update_layout(
        title=f'Matriz de Confusión - Round {last_entry["round"]}',
        xaxis_title='Clase Real',
        yaxis_title='Clase Predicha',
        annotations=annotations,
        template='plotly_white',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_false_negatives_plot(history):
    """Gráfica de falsos negativos y positivos por round (solo para clasificación binaria)"""
    if not history:
        return None
    
    # Filtrar solo entradas de FL training
    fl_entries = [entry for entry in history if entry.get('phase') == 'fl_training' and 'global_metrics' in entry]
    
    if not fl_entries:
        return None
    
    # Verificar si las métricas binarias (fn, fp) existen
    if 'fn' not in fl_entries[0]['global_metrics'] or 'fp' not in fl_entries[0]['global_metrics']:
        # No es clasificación binaria, retornar None
        return None
    
    rounds = [entry['round'] for entry in fl_entries]
    fn = [entry['global_metrics']['fn'] for entry in fl_entries]
    fp = [entry['global_metrics']['fp'] for entry in fl_entries]
    
    fig = go.Figure()
    
    # Falsos Negativos (CRÍTICO)
    fig.add_trace(go.Scatter(
        x=rounds, y=fn,
        mode='lines+markers',
        name='Falsos Negativos (FN)',
        line=dict(color='#d62728', width=3, dash='solid'),
        marker=dict(size=10, symbol='x'),
        fill='tozeroy'
    ))
    
    # Falsos Positivos
    fig.add_trace(go.Scatter(
        x=rounds, y=fp,
        mode='lines+markers',
        name='Falsos Positivos (FP)',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title='Errores Críticos por Round',
        xaxis_title='Round',
        yaxis_title='Cantidad de Casos',
        hovermode='x unified',
        template='plotly_white',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                x=0.5, y=0.95,
                xref='paper', yref='paper',
                text='<i>⚠️ FN = Cáncer NO detectado (MÁS PELIGROSO)</i>',
                showarrow=False,
                font=dict(size=10, color='red')
            )
        ]
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_hospital_evolution_plot(history):
    """Gráfica de evolución de accuracy por hospital"""
    if not history or len(history) < 2:
        return None
    
    # Extraer datos
    rounds = []
    hosp0_acc = []
    hosp1_acc = []
    hosp2_acc = []
    global_acc = []
    
    for entry in history:
        if entry['round'] > 0 and entry.get('phase') == 'fl_training' and 'client_metrics' in entry:
            rounds.append(entry['round'])
            global_acc.append(entry['global_metrics']['accuracy'] * 100)
            
            for client in entry['client_metrics']:
                # Manejar diferentes estructuras de JSON (cancer vs ultrasound)
                if 'metrics' in client:
                    acc = client['metrics']['accuracy'] * 100
                else:
                    acc = client.get('accuracy', 0) * 100
                    
                if client['client_id'] == 0:
                    hosp0_acc.append(acc)
                elif client['client_id'] == 1:
                    hosp1_acc.append(acc)
                elif client['client_id'] == 2:
                    hosp2_acc.append(acc)
    
    fig = go.Figure()
    
    # Hospital 0
    fig.add_trace(go.Scatter(
        x=rounds, y=hosp0_acc,
        mode='lines+markers',
        name='Hospital 0',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Hospital 1
    fig.add_trace(go.Scatter(
        x=rounds, y=hosp1_acc,
        mode='lines+markers',
        name='Hospital 1',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Hospital 2
    fig.add_trace(go.Scatter(
        x=rounds, y=hosp2_acc,
        mode='lines+markers',
        name='Hospital 2',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    # Modelo Global
    fig.add_trace(go.Scatter(
        x=rounds, y=global_acc,
        mode='lines+markers',
        name='Modelo Global',
        line=dict(color='black', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title='Evolución de Accuracy: Hospitales vs Global',
        xaxis_title='Round',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[70, 105]),
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_timing_plot(history):
    """Gráfica de tiempos de entrenamiento FL vs Fine-tuning por hospital"""
    if not history or len(history) == 0:
        return None
    
    # Buscar entry con fine-tuning
    ft_entry = None
    for entry in reversed(history):
        if entry.get('phase') == 'fine_tuning' and 'refined_models' in entry:
            ft_entry = entry
            break
    
    if not ft_entry:
        return None
    
    # Extraer tiempos de FL (acumulativos por hospital)
    fl_times_by_hospital = {}
    for entry in history:
        if entry['round'] > 0 and 'client_metrics' in entry:
            for client in entry['client_metrics']:
                cid = f"Hospital {client['client_id']}"
                if cid not in fl_times_by_hospital:
                    fl_times_by_hospital[cid] = 0
                fl_times_by_hospital[cid] += client.get('fl_training_time', 0)
    
    # Extraer tiempos de fine-tuning
    ft_times = {}
    for refined in ft_entry['refined_models']:
        cid = f"Hospital {refined['client_id']}"
        ft_times[cid] = refined.get('finetune_time', 0)
    
    hospitals = sorted(fl_times_by_hospital.keys())
    fl_times = [fl_times_by_hospital[h] for h in hospitals]
    ft_times_list = [ft_times.get(h, 0) for h in hospitals]
    
    fig = go.Figure()
    
    # FL Training times
    fig.add_trace(go.Bar(
        name='FL Training',
        x=hospitals,
        y=fl_times,
        marker_color='#1f77b4',
        text=[f'{v:.2f}s' for v in fl_times],
        textposition='outside'
    ))
    
    # Fine-tuning times
    fig.add_trace(go.Bar(
        name='Fine-tuning',
        x=hospitals,
        y=ft_times_list,
        marker_color='#ff7f0e',
        text=[f'{v:.2f}s' for v in ft_times_list],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='⏱️ Tiempos de Entrenamiento por Hospital',
        xaxis_title='Hospital',
        yaxis_title='Tiempo (segundos)',
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_refined_comparison_plot(history):
    """Gráfica comparando Modelo Global vs Modelos Refinados"""
    if not history or len(history) == 0:
        return None
    
    # Buscar entry con fine-tuning y el último entry de fl_training para el modelo global
    ft_entry = None
    last_fl_entry = None
    
    for entry in history:
        if entry.get('phase') == 'fl_training' and 'global_metrics' in entry:
            last_fl_entry = entry
        elif entry.get('phase') == 'fine_tuning' and 'refined_models' in entry:
            ft_entry = entry
            
    if not ft_entry or not last_fl_entry:
        return None
    
    # Modelo Global (sin fine-tuning)
    gm = last_fl_entry['global_metrics']
    
    models = ['Modelo Global\n(sin FT)']
    accuracy = [gm['accuracy'] * 100]
    
    # Manejar diferentes estructuras para métricas globales
    if 'sensitivity' in gm:
        sensitivity = [gm['sensitivity'] * 100]
        specificity = [gm['specificity'] * 100]
    elif 'class_metrics' in gm:
        classes = gm['class_metrics']
        avg_recall = sum(c.get('recall', 0) for c in classes.values()) / len(classes) if classes else 0
        sensitivity = [avg_recall * 100]
        specificity = [0]
    else:
        sensitivity = [0]
        specificity = [0]
    
    # Modelos Refinados (con fine-tuning)
    for refined in sorted(ft_entry['refined_models'], key=lambda x: x['client_id']):
        models.append(f"Hospital {refined['client_id']}\n(con FT)")
        
        # Manejar diferentes estructuras de JSON (cancer vs ultrasound)
        if 'refined_metrics' in refined:
            rm = refined['refined_metrics']
            accuracy.append(rm['accuracy'] * 100)
            sensitivity.append(rm.get('sensitivity', 0) * 100)
            specificity.append(rm.get('specificity', 0) * 100)
        else:
            accuracy.append(refined.get('accuracy', 0) * 100)
            if 'class_metrics' in refined:
                classes = refined['class_metrics']
                avg_recall = sum(c.get('recall', 0) for c in classes.values()) / len(classes) if classes else 0
                sensitivity.append(avg_recall * 100)
                specificity.append(0)
            else:
                sensitivity.append(0)
                specificity.append(0)
    
    fig = go.Figure()
    
    # Accuracy
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models,
        y=accuracy,
        marker_color='#1f77b4',
        text=[f'{v:.2f}%' for v in accuracy],
        textposition='outside'
    ))
    
    # Sensitivity
    fig.add_trace(go.Bar(
        name='Sensitivity',
        x=models,
        y=sensitivity,
        marker_color='#ff7f0e',
        text=[f'{v:.2f}%' for v in sensitivity],
        textposition='outside'
    ))
    
    # Specificity
    fig.add_trace(go.Bar(
        name='Specificity',
        x=models,
        y=specificity,
        marker_color='#2ca02c',
        text=[f'{v:.2f}%' for v in specificity],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='🔬 Comparación: Modelo Global vs Modelos Refinados con Fine-tuning',
        yaxis_title='Porcentaje (%)',
        yaxis=dict(range=[0, 110]),
        barmode='group',
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                x=0.5, y=0.95,
                xref='paper', yref='paper',
                text='<i>FT = Fine-tuning con datos adicionales exclusivos de cada hospital</i>',
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def extract_gpu_metrics(history):
    """Extrae métricas de GPU del historial de entrenamiento"""
    if not history:
        return {'available': False}
    
    gpu_available = False
    device_name = 'CPU'
    peak_memory_mb = 0
    avg_memory_mb = 0
    total_memory_mb = 0
    
    memory_samples = []
    
    # Buscar métricas de GPU en el historial
    for entry in history:
        if entry.get('phase') == 'fl_training' and 'hospitals' in entry:
            for hospital in entry['hospitals']:
                if 'gpu_metrics' in hospital:
                    gpu_data = hospital['gpu_metrics']
                    if gpu_data.get('end', {}).get('available', False):
                        gpu_available = True
                        device_name = gpu_data['end'].get('device_name', 'Unknown GPU')
                        total_memory_mb = gpu_data['end'].get('memory_total_mb', 0)
                        peak = gpu_data.get('peak_memory_mb', 0)
                        if peak > 0:
                            memory_samples.append(peak)
                            peak_memory_mb = max(peak_memory_mb, peak)
        
        if entry.get('phase') == 'fine_tuning' and 'refined_models' in entry:
            for model in entry['refined_models']:
                if 'gpu_metrics' in model:
                    gpu_data = model['gpu_metrics']
                    if gpu_data.get('end', {}).get('available', False):
                        gpu_available = True
                        device_name = gpu_data['end'].get('device_name', 'Unknown GPU')
                        total_memory_mb = gpu_data['end'].get('memory_total_mb', 0)
                        peak = gpu_data.get('peak_memory_mb', 0)
                        if peak > 0:
                            memory_samples.append(peak)
                            peak_memory_mb = max(peak_memory_mb, peak)
    
    if memory_samples:
        avg_memory_mb = sum(memory_samples) / len(memory_samples)
    
    return {
        'available': gpu_available,
        'device_name': device_name,
        'peak_memory_mb': round(peak_memory_mb, 2),
        'avg_memory_mb': round(avg_memory_mb, 2),
        'total_memory_mb': round(total_memory_mb, 2),
        'utilization_percent': round((peak_memory_mb / total_memory_mb * 100) if total_memory_mb > 0 else 0, 2)
    }

@app.route('/')
def index():
    """Página principal del dashboard"""
    return render_template('dashboard.html', 
                          refresh_interval=REFRESH_INTERVAL,
                          demos_config=DEMOS_CONFIG)

@app.route('/api/stats')
def api_stats():
    """API para obtener estadísticas actuales"""
    demo_type = request.args.get('demo', 'cancer')
    history = load_training_history(demo_type)
    
    if not history or len(history) == 0:
        return jsonify({
            'status': 'no_data',
            'message': 'No se ha ejecutado ningún entrenamiento aún',
            'demo_type': demo_type
        })
    
    last_entry = history[-1]
    initial_entry = history[0]
    gm = last_entry['global_metrics']
    metrics_type = DEMOS_CONFIG[demo_type]['metrics_type']
    
    # Calcular mejora
    improvement = {
        'accuracy': (gm['accuracy'] - initial_entry['global_metrics']['accuracy']) * 100
    }
    
    # Para métricas binarias
    if metrics_type == 'binary':
        improvement['sensitivity'] = (gm['sensitivity'] - initial_entry['global_metrics']['sensitivity']) * 100
        improvement['specificity'] = (gm['specificity'] - initial_entry['global_metrics']['specificity']) * 100
    
    # Extraer tiempos de entrenamiento
    timing_info = {
        'fl_training_time': 0,
        'finetune_time': 0,
        'total_time': 0
    }
    
    # Buscar entry con fine-tuning para obtener tiempos
    for entry in reversed(history):
        if entry.get('phase') == 'fine_tuning' and 'timing' in entry:
            timing = entry['timing']
            timing_info['fl_training_time'] = round(timing.get('fl_training_time', 0), 2)
            timing_info['finetune_time'] = round(timing.get('finetune_total_time', 0), 2)
            timing_info['total_time'] = round(timing_info['fl_training_time'] + timing_info['finetune_time'], 2)
            break
    
    # Preparar métricas según el tipo
    global_metrics_data = {
        'accuracy': round(gm['accuracy'] * 100, 2)
    }
    
    if metrics_type == 'binary':
        global_metrics_data.update({
            'sensitivity': round(gm['sensitivity'] * 100, 2),
            'specificity': round(gm['specificity'] * 100, 2),
            'tp': gm.get('tp', 0),
            'tn': gm.get('tn', 0),
            'fp': gm.get('fp', 0),
            'fn': gm.get('fn', 0)
        })
    else:
        # Para multiclase, incluir métricas por clase
        if 'class_metrics' in gm:
            global_metrics_data['class_metrics'] = gm['class_metrics']
    
    # Extraer métricas de GPU si están disponibles
    gpu_metrics = extract_gpu_metrics(history)
    
    return jsonify({
        'status': 'ok',
        'demo_type': demo_type,
        'metrics_type': metrics_type,
        'current_round': last_entry['round'],
        'total_rounds': len(history) - 1,
        'phase': last_entry.get('phase', 'fl_training'),
        'global_metrics': global_metrics_data,
        'improvement': improvement,
        'timing': timing_info,
        'gpu_metrics': gpu_metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/plots')
def api_plots():
    """API para obtener todas las gráficas"""
    demo_type = request.args.get('demo', 'cancer')
    history = load_training_history(demo_type)
    
    if not history:
        return jsonify({'status': 'no_data', 'demo_type': demo_type})
    
    metrics_type = DEMOS_CONFIG[demo_type]['metrics_type']
    
    return jsonify({
        'status': 'ok',
        'demo_type': demo_type,
        'plots': {
            'convergence': create_convergence_plot(history, metrics_type),
            'comparison': create_comparison_plot(history),
            'confusion_matrix': create_confusion_matrix_plot(history) if metrics_type == 'binary' else None,
            'false_negatives': create_false_negatives_plot(history) if metrics_type == 'binary' else None,
            'hospital_evolution': create_hospital_evolution_plot(history),
            'timing': create_timing_plot(history),
            'refined_comparison': create_refined_comparison_plot(history)
        }
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🏥 DASHBOARD FLASK - BREAST CANCER FEDERATED LEARNING")
    print("="*70)
    print("\n📊 Dashboard disponible en: http://127.0.0.1:5000")
    print("📡 Actualización automática cada 5 segundos")
    print("\n⚠️  Presiona Ctrl+C para detener el servidor\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
