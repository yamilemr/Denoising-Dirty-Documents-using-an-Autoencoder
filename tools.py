import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(ruta_carpeta, width, height):
    '''
    Carga y preprocesa imágenes de una carpeta
    
    Args:
        folder_path (str): Ruta a la carpeta con imágenes
        width (int): Ancho para redimensionar
        height (int): Alto para redimensionar
        
    Returns:
        np.array: Array de imágenes preprocesadas en escala de grises y normalizadas
    '''
    images = []

    for imagen in os.listdir(ruta_carpeta):
        img = cv2.imread(ruta_carpeta + imagen)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (width, height))
        normalized_img = resized_img / 255.0

        images.append(normalized_img)

    images = np.array(images)[..., np.newaxis]

    return images


def plot_image_pairs_comparison(images1, images2, titles=('Imagen 1', 'Imagen 2'), height=400, width=400):
    '''
    Muestra la comparación de pares de imágenes
    
    Args:
        images1 (np.array): Primer conjunto de imágenes
        images2 (np.array): Segundo conjunto de imágenes
        titles (tuple): Títulos para los conjuntos de imágenes
        height (int): Alto de las imágenes
        width (int): Ancho de las imágenes
    '''
    n = 5  
    indices = np.random.choice(len(images1), size=n, replace=False)

    plt.figure(figsize=(20, 7))

    for i, idx in enumerate(indices):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images1[idx].reshape(height, width), cmap='gray')
        plt.title(titles[0])
        plt.axis('off')

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(images2[idx].reshape(height, width), cmap='gray')
        plt.title(titles[1])
        plt.axis('off')

    plt.show()
    

def plot_noisy_clean_reconstructed(noisy, clean, autoencoder, num_images=5):
    '''
    Muestra una cuadrícula con imágenes ruidosas, limpias y reconstruidas.
    
    Args:
        noisy (np.array): Imágenes con ruido
        clean (np.array): Imágenes limpias (target)
        autoencoder (Model): Modelo de autoencoder entrenado
        num_images (int): Número de ejemplos a mostrar
    '''
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))
    titles = ['Ruidosa', 'Limpia', 'Reconstruida']

    #Selección de índices aleatorios
    random_indices = np.random.choice(len(noisy), num_images, replace=False)

    #Generación de predicciones para los índices aleatorios seleccionados
    predicted_random = autoencoder.predict(noisy[random_indices])

    for i, idx in enumerate(random_indices):
        axes[i, 0].imshow(noisy[idx].squeeze(), cmap='gray')
        axes[i, 1].imshow(clean[idx].squeeze(), cmap='gray')
        axes[i, 2].imshow(predicted_random[i].squeeze(), cmap='gray')

        if i == 0:
            for j in range(3):
                axes[i, j].set_title(titles[j], fontsize=12)

        for j in range(3):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
