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
└── regresion/
    ├── regresion.ipynb
    └── UTKFace/          ← carpeta con las imágenes .jpg
```

> **Nota:** El dataset es de uso exclusivo para investigación no comercial, según los términos de licencia de UTKFace.

---

## Ejecución

```bash
jupyter notebook
# o
jupyter lab
```

Abrir y ejecutar los notebooks en orden:

1. `clasificacion/clasificacion.ipynb`
2. `regresion/regresion.ipynb`

> Cada notebook está diseñado para ejecutarse de principio a fin de forma secuencial. Se recomienda usar **GPU** para el notebook de regresión (entrenamiento CNN).

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

---

## Criterios de Evaluación

### Clasificación

| Criterio | Peso |
|----------|------|
| Justificación teórica y análisis preliminar | 20% |
| Calidad del EDA, feature engineering e interpretaciones | 20% |
| Preprocesamiento y pipeline | 20% |
| Implementación, hiperparámetros y comparación de modelos | 25% |
| Prueba con muestra artificial y análisis | 15% |

### Regresión

| Criterio | Peso |
|----------|------|
| Justificación teórica y análisis preliminar | 20% |
| Calidad del EDA e interpretaciones | 20% |
| Preprocesamiento y pipeline | 20% |
| Implementación y evaluación del modelo CNN | 25% |
| Prueba con muestra artificial y análisis | 15% |

---

> **Nota:** Este workshop será sustentado por un miembro del equipo seleccionado aleatoriamente. El entendimiento individual de cada integrante es fundamental.