import os
from datasets import load_dataset, load_from_disk, Dataset
from collections import Counter
import json
from config import DATASET_PATH, NUM_CLASSES
from PIL import Image
import numpy as np


#  === Core of this code: ===
#  1) Download the PACS dataset from Hugging Face
#  2) Automatically selects the 4 most balanced classes
#  3) Creates a balanced subset
#  4) Save the dataset locally (first time only)
#  5) In subsequent runs, load the already filtered dataset directly


def get_balanced_subset(dataset, num_classes=4):
    """
    Crea un sottoinsieme bilanciato del dataset, selezionando le classi più bilanciate
    e rimappando le etichette scelte a nuovi indici consecutivi (es. 0, 1, 2, 3).

    Args:
        dataset (Dataset): Dataset originale.
        num_classes (int): Numero di classi da selezionare.

    Returns:
        tuple: (
            Dataset bilanciato con nuove label,
            etichette originali selezionate,
            nomi delle classi,
            dizionario di rimappatura (originale → nuova label)
        )
    """
    # Estrae tutte le etichette (numeriche) dal dataset
    labels = dataset["label"]

    # Contiene i nomi leggibili delle classi (es. "dog", "giraffe", etc.)
    label_names = dataset.features["label"].names

    # Usa Counter per contare quante volte appare ogni etichetta
    # Risultato: {0: 1200, 1: 1250, 2: 300, 3: 298, 4: 700, ...}
    label_counts = Counter(labels)

    # Seleziona le classi con meno variazione nel numero di esempi (più bilanciate)
    # label_counts.items() restituisce (etichetta, numero di immagini)
    # Ordinamento in base alla quantità di immagini (x[1])
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1])

    # Prendiamo solo gli ID delle etichette con meno immagini
    # Es: se num_classes = 4, magari otteniamo [3, 2, 0, 5]
    selected_labels = [idx for idx, _ in sorted_counts[:num_classes]]
    print("Classi selezionate:", [label_names[i] for i in selected_labels])

    # Filtra dataset
    def filter_classes(example):
        # Ritorna True solo se l’etichetta dell’esempio è tra quelle selezionate
        return example["label"] in selected_labels

    filtered = dataset.filter(filter_classes)

    # Raggruppa per etichetta
    # Crea un dizionario con chiavi = etichette e valori = lista di esempi
    grouped = {label: [] for label in selected_labels}
    for ex in filtered:
        # Raggruppa tutti gli esempi per la loro etichetta (es. tutte le giraffe insieme, ecc.)
        grouped[ex["label"]].append(ex)

    # Uniforma il numero di immagini per classe
    # Serve per tagliare tutte le classi allo stesso numero (quella che ne ha di meno è il limite)
    min_len = min(len(exs) for exs in grouped.values())
    print(f"Numero immagini per classe: {min_len}")

    # Rimappa le etichette originali a nuovi indici consecutivi
    label_mapping = {old_label: new_idx for new_idx, old_label in enumerate(selected_labels)}

    # Crea il sottoinsieme bilanciato e rimappa le etichette
    balanced_samples = []
    for label, exs in grouped.items():
        for ex in exs[:min_len]:
            ex["label"] = label_mapping[label]
            balanced_samples.append(ex)

    # Costruisci il nuovo Dataset
    balanced_dataset = Dataset.from_list(balanced_samples)

    # Ritorna anche le etichette scelte e i nomi delle classi
    return balanced_dataset, selected_labels, label_names, label_mapping


def save_class_info(path, selected_labels, label_names, label_mapping):
    """
    Salva le informazioni sulle classi in un file JSON.

    Args:
        path (str): Percorso del file.
        selected_labels (list): Etichette selezionate (originali).
        label_names (list): Nomi delle classi.
        label_mapping (dict): Mappa da etichetta originale a nuova etichetta.
    """
    with open(os.path.join(path, "class_info.json"), "w") as f:
        json.dump({
            "selected_labels": selected_labels,
            "label_names": label_names,
            "label_mapping": {str(k): v for k, v in label_mapping.items()}  # serializzabile in JSON
        }, f)


def load_class_info(path):
    """
    Carica le informazioni sulle classi da un file JSON.

    Returns:
        tuple: (etichette selezionate, nomi delle classi, mappa delle etichette).
    """
    with open(os.path.join(path, "class_info.json"), "r") as f:
        info = json.load(f)
        return info["selected_labels"], info["label_names"], {
            int(k): v for k, v in info.get("label_mapping", {}).items()
        }


def save_images_from_dataset(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    paths = []

    for idx, example in enumerate(dataset):
        img_array = np.array(example["image"])
        img = Image.fromarray(img_array)
        img_path = os.path.join(save_dir, f"img_{idx}.png")
        img.save(img_path)
        paths.append(img_path)

    dataset = dataset.add_column("file", paths)
    return dataset


def main():
    """
    Prepara il dataset bilanciato. Se il dataset esiste già, lo carica.
    Altrimenti, lo scarica, lo bilancia e lo salva.
    """
    if os.path.exists(DATASET_PATH):
        print(f"✔️ Dataset già salvato trovato in '{DATASET_PATH}'. Lo carico...")
        balanced_dataset = load_from_disk(DATASET_PATH)

        # Carica le info sulle classi selezionate
        selected_labels, label_names = load_class_info(DATASET_PATH)
        selected_class_names = [label_names[i] for i in selected_labels]
        print(f"Classi selezionate: {selected_class_names}")

    else:
        print("⬇️ Scarico PACS dataset da Hugging Face...")

        # Se non esiste ancora, allora lo scarica da Hugging Face
        dataset = load_dataset("flwrlabs/pacs", split="train")

        # Ottieni il sottoinsieme bilanciato e le informazioni sulle classi
        balanced_dataset, selected_labels, label_names, label_mapping = get_balanced_subset(
            dataset, NUM_CLASSES
        )

        print(balanced_dataset[0])

        for original, new in label_mapping.items():
            print(f"{label_names[original]} → {new}")

        # Mostra le classi selezionate
        selected_class_names = [label_names[i] for i in selected_labels]
        print(f"Classi selezionate: {selected_class_names}")

        print(f"Salvo il sottoinsieme bilanciato in '{DATASET_PATH}'...")
        balanced_dataset.save_to_disk(DATASET_PATH)

        # Salva le info delle classi in un file JSON
        save_class_info(DATASET_PATH, selected_labels, label_names, label_mapping)

        # Salvataggio delle immagini in una cartella
        image_dir = os.path.join(DATASET_PATH, "images")
        balanced_dataset = save_images_from_dataset(balanced_dataset, image_dir)

    print(
        f"✅ Dataset pronto con {len(balanced_dataset)} immagini bilanciate tra {NUM_CLASSES} classi."
    )

    # Verifica una delle immagini
    sample = balanced_dataset[0]
    label_index = sample["label"]
    img_path = sample["file"]
    label_name = selected_class_names[label_index]

    print("\n🔎 Verifica immagine:")
    print(f"📍 Percorso immagine: {img_path}")
    print(f"🏷️ Label (indice): {label_index}")
    print(f"🧾 Label (nome): {label_name}")


if __name__ == "__main__":
    main()
