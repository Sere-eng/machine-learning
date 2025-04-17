import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
from config import DATASET_PATH


def apply_pca(X, n_components):
    """
    Applica PCA ai dati. Se n_components < 0, usa le ultime |n| componenti.
    
    Args:
        X (np.ndarray): dati (campioni x features)
        n_components (int): numero di componenti. Se negativo, prende le ultime.
        
    Returns:
        (X_pca, pca): dati proiettati e oggetto PCA per inverse_transform
    """
    if n_components < 0:
        # PCA completa
        full_pca = PCA()
        X_full_pca = full_pca.fit_transform(X)

        # Ultime |n_components| componenti
        X_pca = X_full_pca[:, n_components:]
        
        # Ricreiamo un oggetto PCA con solo quelle componenti
        pca = PCA(n_components=abs(n_components))
        pca.components_ = full_pca.components_[n_components:]
        pca.mean_ = full_pca.mean_
        pca.n_components_ = abs(n_components)
    else:
        # PCA normale con le prime n componenti
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
    
    return X_pca, pca


def flatten_image(img):
    """
    Appiattisce un'immagine in un array monodimensionale e restituisce anche la sua forma originale.

    Args:
        img (numpy.ndarray): Immagine da appiattire.

    Returns:
        numpy.ndarray: Immagine appiattita.
    """
    return img.ravel(), img.shape


def reconstruct_image(pca, X_pca):
    """
    Ricostruisce i dati originali dallo spazio delle componenti principali.

    Args:
        pca (PCA): Oggetto PCA utilizzato per la trasformazione.
        X_pca (numpy.ndarray): Dati nello spazio delle componenti principali.

    Returns:
        numpy.ndarray: Dati ricostruiti nello spazio originale.
    """
    return pca.inverse_transform(X_pca)


def save_image(arr, path, shape):
    """
    Salva un'immagine da un array.

    Args:
        arr (numpy.ndarray): Array dell'immagine.
        path (str): Percorso dove salvare l'immagine.
    """
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr.reshape((shape)))
    img.save(path)


def load_images_and_labels(dataset):
    """
    Carica immagini e label e shape originali dal dataset.

    Args:
        dataset: Dataset Hugging Face.

    Returns:
        tuple: (matrice X, vettore y).
    """
    image_dir = os.path.join(DATASET_PATH, "images")  # usa il path dal config
    X = []
    y = []
    shapes = []
    for i, example in enumerate(dataset):
        image_path = os.path.join(image_dir, f"img_{i}.png")
        img = np.array(Image.open(image_path))
        flat_img, shape = flatten_image(img)
        X.append(flat_img)
        shapes.append(shape)
        y.append(example["label"])
    return np.array(X), np.array(y), shapes

