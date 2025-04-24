import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datasets import load_from_disk
from pca_utils import load_images_and_labels, apply_pca
from config import DATASET_PATH, OUTPUT_DIR
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


def plot_decision_boundary(X, y, clf, title, ax, xlabel, ylabel, class_names):
    """
    Traccia i confini di decisione del classificatore.
    Mostra anche le etichette delle classi nella legenda.

    Args:
        X (np.ndarray): Dati di input ridotti tramite PCA.
        y (np.ndarray): Etichette dei dati.
        clf (sklearn classifier): Classificatore addestrato.
        title (str): Titolo del grafico.
        ax (matplotlib axis): Asse su cui disegnare.
        xlabel (str): Etichetta asse x.
        ylabel (str): Etichetta asse y.
        class_names (list): Nomi leggibili delle classi.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Qui si sta dicendo: "Fammi la griglia in base alla prima colonna e alla seconda colonna dell’array che ricevo", quindi possono essere le prime due componenti principali o le terze e quarte, ecc.
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Creazione scatter plot con legenda leggibile
    scatter_plot = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    legend_labels = [f"{i}: {name}" for i, name in enumerate(class_names)]
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=scatter_plot.cmap(scatter_plot.norm(i)),
               label=legend_labels[i]) for i in range(len(class_names))]
    ax.legend(handles=handles, title="Classi")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def main():
    # Caricamento del dataset
    dataset = load_from_disk(DATASET_PATH)

    # Caricamento immagini e label
    X_raw, y, shapes = load_images_and_labels(dataset)
    
    # Caricamento dei nomi delle classi
    class_names = load_class_names()

    # Standardizzazione: zero-mean e unit variance -> Fondamentale per la PCA
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Applicazione della PCA con 60 componenti principali
    X_train_pca, pca = apply_pca(X_train, 60)
    X_test_pca = pca.transform(X_test)

    # Estrazione delle componenti per i vari esperimenti
    X_train_pc12 = X_train_pca[:, 0:2]  # PC1 e PC2
    X_test_pc12 = X_test_pca[:, 0:2]

    X_train_pc34 = X_train_pca[:, 2:4]  # PC3 e PC4
    X_test_pc34 = X_test_pca[:, 2:4]

    X_train_pc60 = X_train_pca[:, 0:60]  # tutte le prime 60 componenti
    X_test_pc60 = X_test_pca[:, 0:60]

    # === Classificazione con Naive Bayes ===
    clf_pc12 = GaussianNB().fit(X_train_pc12, y_train)
    clf_pc34 = GaussianNB().fit(X_train_pc34, y_train)
    clf_pc60 = GaussianNB().fit(X_train_pc60, y_train)

    # === Predizioni ===
    y_pred_pc12 = clf_pc12.predict(X_test_pc12)
    y_pred_pc34 = clf_pc34.predict(X_test_pc34)
    y_pred_pc60 = clf_pc60.predict(X_test_pc60)

    # === Accuratezze ===
    acc_pc12 = accuracy_score(y_test, y_pred_pc12)
    acc_pc34 = accuracy_score(y_test, y_pred_pc34)
    acc_pc60 = accuracy_score(y_test, y_pred_pc60)

    # === Stampa delle performance ===
    print(f"Accuracy con PC1 & PC2: {acc_pc12:.4f}")
    print(f"Accuracy con PC3 & PC4: {acc_pc34:.4f}")
    print(f"Accuracy con tutte le 60 componenti: {acc_pc60:.4f}")

    # Visualizzazione dei confini di decisione
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot decision boundary per PC1 e PC2
    plot_decision_boundary(
        X_train_pc12, y_train, clf_pc12,
        "Decision Boundary: PC1 & PC2", axes[0], "PC1", "PC2", class_names)

    # Plot decision boundary per PC3 e PC4
    plot_decision_boundary(
        X_train_pc34, y_train, clf_pc34,
        "Decision Boundary: PC3 & PC4", axes[1], "PC3", "PC4", class_names)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_boundaries.png"))
    plt.show()

    # === Grafico comparativo delle accuracy ===
    plt.figure(figsize=(6, 4))
    plt.bar(['PC1&2', 'PC3&4', 'PC1-60'], [acc_pc12, acc_pc34, acc_pc60], color=['blue', 'green', 'orange'])
    plt.title("Comparazione delle Accuratezze")
    plt.xlabel("Componenti Principali")
    plt.ylabel("Accuratezza")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
    plt.show()


if __name__ == "__main__":
    main()
