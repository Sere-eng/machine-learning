# === Percorsi ===
# Path dove è salvato il dataset bilanciato
DATASET_PATH = "pacs_balanced_4class"

# Path di output per immagini proiettate con PCA
OUTPUT_DIR = "pca_outputs"

# === PCA Settings ===
# Numero di componenti principali da testare
PCA_COMPONENTS = [60, 6, 2, -6]  # -6 = ultime 6 componenti

# Numero di classi da usare (es: 4 per PACS)
NUM_CLASSES = 4
