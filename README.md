# Workshop 2 — Machine Learning & Deep Learning Aplicado

> **Universidad EAFIT · Introducción a la Inteligencia Artificial · 2026-01**

Implementación de dos problemas supervisados independientes que cubren el ciclo completo de un proyecto de ML/DL: exploración de datos, preprocesamiento, entrenamiento, evaluación y análisis crítico de resultados.

| Problema | Tipo | Dataset |
|----------|------|---------|
| Detección de fatiga muscular en ciclismo | Clasificación binaria | HuggingFace — `YominE/Muscle_Fatigue_Cycling` |
| Estimación de edad a partir de imágenes faciales | Regresión | Kaggle — `arashnic/faces-age-detection-dataset` |

**Equipo:** Daniela Salazar · Laura Indabur · Athina Cappelletti

---

## Estructura del Repositorio

```
workshop2_ia/
├── README.md
├── requirements.txt
├── .gitignore
├── clasificacion/
│   └── clasificacion.ipynb      ← Notebook completo — Problema 1
└── regresion/
    └── regresion.ipynb          ← Notebook completo — Problema 2
```

---

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/dsalazar444/workshop2_ia.git
cd workshop2_ia
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Las dependencias incluyen:

```
numpy · pandas · scikit-learn · scipy
tensorflow · torch · torchvision
datasets · transformers · pyarrow
matplotlib · seaborn
kagglehub · tqdm
```

---

## Descarga de Datasets

### Problema 1 — Muscle Fatigue Cycling (HuggingFace)

El dataset se descarga automáticamente dentro del notebook usando la librería `datasets`. No se requiere configuración adicional.

```python
from datasets import load_dataset
dataset = load_dataset("YominE/Muscle_Fatigue_Cycling")
```

### Problema 2 — UTKFace Dataset (descarga manual)

El dataset utilizado es **UTKFace**, un conjunto de más de 20.000 imágenes faciales etiquetadas por edad (0–116 años), género y etnicidad. La etiqueta de edad está embebida directamente en el nombre de cada archivo con el formato `[age]_[gender]_[race]_[datetime].jpg`.

**Pasos para descargar:**

1. Ir a [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)
2. Descargar la versión **Aligned & Cropped Faces** (ZIP · 107 MB) desde Google Drive
3. Extraer el contenido y ubicar la carpeta de imágenes en el directorio del notebook:

```
workshop2_ia/
└── clasificacion/
    ├── clasificacion.ipynb
└── regresion/
    ├── regresion.ipynb
    ├── regresion_eda.ipynb    ← Análisis Exploratorio de Datos del problema
    ├── data_split.ipynb       ← ⚠️ EJECUTAR PRIMERO
    ├── data/                  ← carpeta con las imágenes .jpg (extraídas)
    │   ├── part1/
    │   ├── part2/
    │   └── part3/
    └── dataset/               ← se crea automáticamente tras ejecutar data_split.ipynb
        ├── train/             (70% de imágenes)
        ├── val/               (15% de imágenes)
        └── test/              (15% de imágenes)
```

> **Nota:** El dataset es de uso exclusivo para investigación no comercial, según los términos de licencia de UTKFace.

**⚠️ Paso obligatorio: Dividir el dataset**

Antes de ejecutar `regresion.ipynb`, **debes ejecutar** `data_split.ipynb` para dividir las imágenes en carpetas de train, val y test:

```bash
jupyter notebook regresion/data_split.ipynb
# Ejecutar todas las celdas → genera dataset/ con subdirectorios train/, val/, test/
```

Este notebook:
- Lee todas las imágenes de `data/part1/`, `data/part2/`, `data/part3/`
- Crea las carpetas `dataset/train/`, `dataset/val/`, `dataset/test/`
- Distribuye las imágenes respetando el balance de clases
- Guarda un log en `split_log.csv` con la asignación de cada imagen

**Orden de ejecución obligatorio:**

1. ✅ `data_split.ipynb` — crear dataset/train, dataset/val, dataset/test
2. ✅ `regresion.ipynb` — EDA + entrenamiento CNN

---

## Ejecución

```bash
jupyter notebook
# o
jupyter lab
```

### Problema 1 — Clasificación (Fatiga Muscular)

Abrir y ejecutar:

```
jupyter notebook problema1.ipynb
```

El notebook descarga automáticamente el dataset de HuggingFace. Ejecutar todas las celdas en orden secuencial.

### Problema 2 — Regresión (Estimación de Edad)

**⚠️ IMPORTANTE: Orden de ejecución**

1. **Primero:** Ejecutar `data_split.ipynb` para dividir el dataset:
```bash
jupyter notebook regresion/data_split.ipynb
# Ejecutar todas las celdas
# ✓ Se crean las carpetas: dataset/train/, dataset/val/, dataset/test/
# ✓ Se genera split_log.csv con el historial de la división
```

2. **Segundo:** Ejecutar `regresion.ipynb` para EDA y entrenamiento:
```bash
jupyter notebook regresion/regresion.ipynb
# Ejecutar todas las celdas en orden
```

> El notebook de regresión espera encontrar las carpetas `dataset/train/`, `dataset/val/` y `dataset/test/` ya creadas. Si las carpetas no existen, el notebook fallará.

> Se recomienda usar **GPU** para el notebook de regresión (entrenamiento CNN es más rápido).

---

## Problema 1 — Clasificación: Detección de Fatiga Muscular

Señales EMG registradas en 8 músculos de la pierna dominante durante sprints en bicicleta. Target binario:

| Etiqueta | Significado |
|----------|-------------|
| `0` | Condición normal |
| `1` | Desgaste muscular |

**Pipeline:**

- Feature engineering con ventanas de 1 segundo sobre 8 canales EMG (características en dominio del tiempo y frecuencia)
- EDA completo: distribuciones, correlaciones, boxplots por clase, balance de clases
- Preprocesamiento con pipeline de `scikit-learn` · Split 70/15/15
- Entrenamiento y comparación de: kNN, Decision Tree, Random Forest, Gradient Boosting, DNN
- Ajuste de hiperparámetros (Grid Search / Random Search)
- Evaluación con Accuracy, Precision, Recall, F1-Score y matriz de confusión
- Prueba con muestra artificial sintética

---

## Problema 2 — Regresión: Estimación de Edad

Imágenes faciales etiquetadas con la edad del sujeto. El modelo estima la edad a partir de los píxeles.

**Pipeline:**

- EDA: distribución de edades, análisis de sesgo, visualización de muestras
- Preprocesamiento: redimensionamiento, normalización, data augmentation · Split 70/15/15
- Modelo CNN con capas convolucionales, Dropout y Batch Normalization
- Función de pérdida: MAE / MSE / Huber Loss
- Métricas: MAE, RMSE, R²
- Análisis de curvas de entrenamiento/validación (overfitting/underfitting)
- Prueba con imagen real y análisis de sensibilidad
