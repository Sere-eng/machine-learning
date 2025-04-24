import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datasets import load_from_disk
from config import DATASET_PATH, PCA_COMPONENTS, OUTPUT_DIR
from pca_utils import apply_pca, save_image, load_images_and_labels, reconstruct_image
import numpy as np
import json


def load_class_names():
    """
    Carica i nomi delle classi dal file class_info.json.
    """
    with open(os.path.join(DATASET_PATH, "class_info.json"), "r") as f:
        info = json.load(f)
        label_names = info["label_names"]
        label_mapping = {int(k): v for k, v in info["label_mapping"].items()}
        return [label_names[k] for k in sorted(label_mapping.keys(), key=lambda x: label_mapping[x])]


def main():
    # Creazione della directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Caricamento del dataset
    dataset = load_from_disk(DATASET_PATH)

    # Caricamento dei nomi delle classi
    class_names = load_class_names()

    # Caricamento immagini e label
    X_raw, y, shapes = load_images_and_labels(dataset)

    # Standardizzazione: zero-mean e unit variance -> Fondamentale per la PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Salva immagine originale (non standardizzata)
    save_image(X_raw[0], os.path.join(OUTPUT_DIR, "original.png"), shapes[0])

    # Applica PCA e salva immagini ricostruite
    for n in PCA_COMPONENTS:
        X_pca, pca = apply_pca(X, n)

        # Ricostruzione dallo spazio PCA
        X_reconstructed = reconstruct_image(pca, X_pca)
        x_reconstructed = scaler.inverse_transform(X_reconstructed[0].reshape(1, -1)).flatten()  # Inverto anche la standardizzazione, per riportarla nel dominio originale

        # Salva immagine ricostruita
        filename = f"reconstructed_{abs(n)}{'last' if n < 0 else ''}.png"
        save_image(x_reconstructed, os.path.join(OUTPUT_DIR, filename), shapes[0])

    # Visualizzazione scatter plot della PCA rispettivamente con:
    # prime due componenti principali (PC1 e PC2)
    # le componenti PC3 e PC4
    # le componenti PC10 e PC11
    X_pca_full, _ = apply_pca(X, 12)  # Prendi almeno 11 componenti

    pairs = [(0, 1), (2, 3), (9, 10)]
    titles = ["PC1 vs PC2", "PC3 vs PC4", "PC10 vs PC11"]

    for (i, j), title in zip(pairs, titles):
        plt.figure(figsize=(10, 8))

        # Scatter plot con colorazione in base alla classe
        scatter = plt.scatter(X_pca_full[:, i], X_pca_full[:, j], c=y, cmap="viridis", alpha=0.7)

        # Colorbar standard
        cbar = plt.colorbar(label="Indice classe")
        # Personalizza la colorbar, ad esempio:
        cbar.set_ticks(range(len(class_names)))
        cbar.set_ticklabels(class_names)

        # Aggiungi la legenda con i nomi delle classi
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=scatter.cmap(scatter.norm(c)),
                                       markersize=10, label=class_names[c])
                            for c in range(len(class_names))],
                   title="Classi")

        plt.title(f"Proiezione PCA ({title})")
        plt.xlabel(f"PC{i+1}")
        plt.ylabel(f"PC{j+1}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(OUTPUT_DIR, f"pca_scatter_PC{i+1}_PC{j+1}.png"), dpi=300)
        plt.close()

    # Grafico della varianza spiegata cumulativa
    pca_full = PCA()
    pca_full.fit(X)

    explained = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure()
    plt.plot(np.arange(1, len(explained) + 1), explained, marker='o')
    plt.xlabel("Numero di componenti principali")
    plt.ylabel("Varianza spiegata cumulata")
    plt.title("Varianza spiegata cumulativa PCA")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "explained_variance.png"))
    plt.close()


if __name__ == "__main__":
    main()
