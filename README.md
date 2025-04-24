# **ELiminación de Ruido de Documentos usando un Autoencoder**

Este proyecto implementa un autoencoder convolucional para eliminar el ruido de imágenes de documentos, mejorando su legibilidad y calidad. El modelo se entrena con un conjunto de datos que contiene imágenes ruidosas con sus versiones limpias, con el objetivo de restaurar las imágenes ruidosas.

## Características principales
* **Autoencoder Convolucional:** Arquitectura eficiente para procesamiento de imágenes, compuesta por un encoder y un decoder.
* **Entrenamiento:** Modelo entrenado para minimizar el error cuadrático medio (MSE) entre las imágenes limpias y las reconstruidas.
* **Métricas de evaluación:** Incluye MSE, PSNR y SSIM para medir la calidad de la reconstrucción.
* **Visualización:** Herramientas para comparar imágenes originales, limpias y reconstruidas.

## Resultados
El modelo logra reducir significativamente el ruido en los documentos, manteniendo la estructura y legibilidad del texto. Las métricas obtenidas son:
* **MSE:** 0.358, indicando una reconstrucción precisa.
* **PSNR:** 14.461 dB, reflejando una calidad aceptable.
* **SSIM promedio:** 0.826, conservando la estructura global del documento.

## Archivos Disponibles
* `images`: Carpeta con imágenes de entrenamiento y prueba:
    - `train`: Documentos con ruido (entrenamiento).
    - `train_cleaned`: Documentos limpios (entrenamiento).
    - `test`: Documentos con ruido (prueba).
* `autoencoder_denoising.ipynb`: Notebook principal con el código del proyecto
* `tools.py`: Funciones auxiliares para carga y visualización.
* `document_denoising_autoencoder.h5`: Modelo entrenado guardado.

## Autora
Yamile Montecinos: [@yamilemr](https://github.com/yamilemr)