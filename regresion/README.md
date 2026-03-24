# Regresion - Problema 2

Esta carpeta contiene la parte de regresion del workshop.

## Archivo principal

- `problema2_eda.ipynb`: cubre el análisis preliminar, EDA, procesamiento de datos y preparación del modelo.


## Donde ubicar el dataset

Descarga el dataset de Kaggle `arashnic/faces-age-detection-dataset` y extraelo en una de estas rutas:

- `regresion/data/faces_age_detection/`
- `regresion/data/`


## Alternativa automatica

El notebook `problema2_eda.ipynb` tambien puede intentar descargar el dataset automaticamente con `kagglehub` si no encuentra imagenes en las rutas locales.

## Preparación del dataset

Se creó una carpeta llamada `dataset/` donde se almacenan las imágenes originales del dataset.  
Dentro de esta carpeta se deben ubicar las subcarpetas:

<pre>
dataset/
├── part1/
├── part2/
├── part3/
</pre>

Estas carpetas contienen las imágenes originales en formato UTKFace.

Posteriormente, se utilizó el notebook `data_split.ipynb` para realizar la división del dataset en los conjuntos de entrenamiento, validación y prueba. Este proceso toma las imágenes de `part1`, `part2` y `part3`, y las organiza automáticamente en la siguiente estructura:

```
dataset/
├── train/
├── val/
├── test/
```

Es importante mantener la estructura de `part1`, `part2` y `part3` antes de ejecutar el notebook, ya que de ello depende que la división se realice correctamente.


## Nota

La carpeta `regresion/data/` y `dataset/` esta ignorada en git para evitar subir imagenes del dataset al repositorio.

