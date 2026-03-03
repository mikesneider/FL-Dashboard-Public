# Federated Learning — Results Dashboard

Interactive notebook to visualize Federated Learning training results for breast cancer detection using NVFLARE.

## Demos

| Demo | Type | Description |
|---|---|---|
| **Breast Cancer** | Tabular | Binary classification, 30 features, 569 samples |
| **Breast Ultrasound** | Images | Multiclass classification (benign / malignant / normal), ResNet18 |

## Project Structure

```
FL-Dashboard-Public/
├── notebooks/
│   └── ResultadosFL_Dashboard.ipynb   # Main dashboard notebook
├── scripts/
│   └── python/
│       └── dashboard_flask_cancer.py  # Visualization functions
├── training_history_cancer_nvflare.json      # Cancer training results
├── training_history_ultrasound_nvflare.json  # Ultrasound training results
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter lab notebooks/ResultadosFL_Dashboard.ipynb
```

Select the demo in cell 4:
```python
DEMO_TYPE = 'cancer'      # Breast Cancer (tabular)
DEMO_TYPE = 'ultrasound'  # Breast Ultrasound (images)
```

Then run all cells.

## Datasets

| Dataset | Source | License |
|---|---|---|
| Wisconsin Breast Cancer | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | CC BY 4.0 |
| Breast Ultrasound (BUSI) | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) | Academic use |

**References:**
- Street W.N., Wolberg W.H., Mangasarian O.L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* SPIE Vol. 1905, pp. 861–870.
- Al-Dhabyani W. et al. (2020). *Dataset of breast ultrasound images.* Data in Brief, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863

## What is Federated Learning?

Federated Learning is a machine learning approach where the model is trained across multiple decentralized nodes (e.g., hospitals) **without sharing raw data**. Only model gradients are exchanged, preserving data privacy.

```
Hospital A  ──┐
Hospital B  ──┼──► FL Server (aggregates gradients) ──► Global Model
Hospital C  ──┘
```

## Results Overview

The training history JSON files contain:
- Round-by-round global model metrics (accuracy, sensitivity, specificity)
- Per-hospital performance during training
- Fine-tuning results after local model refinement
- Training and fine-tuning time metrics
- GPU memory usage (if applicable)
