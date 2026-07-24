import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================================================================
# 0. CONFIG
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Public sources matching the paper's Table 2 per-class counts:
#   Mendeley -> Sethy et al. (2020), data.mendeley.com/datasets/fwcj7stb8r/1
#   GitHub   -> github.com/aldrin233/RiceDiseases-DataSet
#   UCI      -> archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases
# The paper's 533 self-collected Primary-data photos (Sec 3.1.1) are not
# publicly available; only a partial subset exists in Kaggle mirrors.
DATA_ROOTS = [
    "/kaggle/input/datasets/soumyasisdas/leafdeasese/dataset",
    "/kaggle/input/datasets/soumyasisdas/newleaf/dataset",
]

# Folders excluded from every source, checked anywhere in the path:
#   - "sample augmented dataset": GAN-output example, not real photos
#   - "rotated": pre-baked rotated duplicates in the GitHub source; only
#     the "Orig" images are used
EXCLUDE_PATH_KEYWORDS = {"sample augmented dataset", "rotated"}

# "auto" scans all configured roots and picks whichever grouping (raw
# Primary+secondary folders, a pre-merged "complete dataset" folder, or
# everything combined) yields the most unique images after de-duplication.
DATA_SOURCE_MODE = "auto"

IMG_SIZE = 64
SEG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50              # Sec 3.5.1's stated optimum
GAN_LATENT_DIM = 100
GAN_EPOCHS = 150

# Sec 3.6 algorithm step 7: GAN-generated images are saved to disk under
# GAN_OUTPUT_DIR/<disease_class>/ before being merged with the dataset.
SAVE_GAN_IMAGES_TO_DISK = True
GAN_OUTPUT_DIR = "/kaggle/working/gan_augmented_dataset"

USE_TRADITIONAL_AUGMENTATION = False
TRADITIONAL_AUG_PER_IMAGE = 3

# Paper's literal per-class GAN augmentation percentages (Sec 3.2).
AUG_MULTIPLIER = {
    "Bacterial_blight": 4.24,
    "Blast": 1.80,
    "Leaf_smut": 9.22,
}

SEVERITY_LEVELS = ["mild", "moderate", "severe", "profound"]
DISEASE_CLASSES = ["Bacterial_blight", "Blast", "Leaf_smut"]

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ============================================================================
# 1. DATA COLLECTION (Sec 3.1)
# ============================================================================
def folder_to_class(folder_name):
    """Maps inconsistently-named folders to the three canonical classes."""
    name = folder_name.lower()
    if "smut" in name:
        return "Leaf_smut"
    if "blight" in name:
        return "Bacterial_blight"
    if "blast" in name:
        return "Blast"
    return None


def folder_to_class_from_path(dirpath, root):
    """Checks the whole path (not just the immediate parent) for a class
    keyword, since some sources nest images one level deeper (e.g.
    blast/Orig/img.jpg)."""
    rel = os.path.relpath(dirpath, root)
    parts = rel.split(os.sep) if rel != "." else []
    for part in reversed(parts):
        cls = folder_to_class(part)
        if cls is not None:
            return cls
    return None


def _scan_all_images(data_roots, exclude_path_keywords):
    """Walks every root, records every image with its class and top-level
    folder, and reports any folders that didn't match a known class."""
    records = []
    unmatched = []
    for root in data_roots:
        if not os.path.isdir(root):
            print(f"[warn] configured data root not found, skipping: {root}")
            continue
        root_label = os.path.basename(os.path.normpath(root))

        for dirpath, _, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            path_parts_lower = [p.lower() for p in rel.split(os.sep)] if rel != "." else []
            if any(kw in part for part in path_parts_lower for kw in exclude_path_keywords):
                continue
            top_level_component = rel.split(os.sep)[0] if rel != "." else "(root)"

            image_files = [f for f in filenames if f.lower().endswith(VALID_EXT)]
            if not image_files:
                continue

            cls = folder_to_class_from_path(dirpath, root)
            top_level = f"{root_label}/{top_level_component}"
            if cls is None:
                unmatched.append((dirpath, len(image_files)))
                continue

            for fname in image_files:
                fpath = os.path.join(dirpath, fname)
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    fsize = -1
                # dedup_key includes `cls` so images from different classes
                # are never treated as duplicates of one another.
                records.append({
                    "filepath": fpath,
                    "top_level": top_level,
                    "disease": cls,
                    "dedup_key": (cls, fname.lower(), fsize),
                })

    if unmatched:
        print("[note] image files found in unclassified folders (excluded):")
        for d, n in unmatched:
            print(f"    {d}   ({n} images)")

    return pd.DataFrame(records)


def _dedup(df):
    if df.empty:
        return df
    return df.drop_duplicates(subset="dedup_key").reset_index(drop=True)


def build_dataframe(data_roots, mode=DATA_SOURCE_MODE, exclude_path_keywords=EXCLUDE_PATH_KEYWORDS):
    if isinstance(data_roots, str):
        data_roots = [data_roots]

    all_df = _scan_all_images(data_roots, exclude_path_keywords)
    if all_df.empty:
        raise RuntimeError(
            f"No images found anywhere under {data_roots}. "
            f"Double-check DATA_ROOTS and that the datasets are attached."
        )

    print("Images found on disk, by root/top-level folder and class:")
    print(pd.crosstab(all_df["top_level"], all_df["disease"], margins=True))
    print()

    top_levels = all_df["top_level"].unique()
    primsec = [t for t in top_levels
               if "primary data" in t.lower() or "secondary data" in t.lower()]
    complete = [t for t in top_levels if "complete dataset" in t.lower()]

    candidates = {}
    if primsec:
        candidates["primary_secondary"] = all_df[all_df["top_level"].isin(primsec)]
    if complete:
        candidates["complete_dataset"] = all_df[all_df["top_level"].isin(complete)]
    candidates["all"] = all_df
    candidates = {k: _dedup(v) for k, v in candidates.items() if not v.empty}

    if mode == "auto":
        chosen_name = max(candidates, key=lambda k: len(candidates[k]))
    else:
        if mode not in candidates:
            raise RuntimeError(
                f"DATA_SOURCE_MODE='{mode}' requested but no images found for "
                f"it. Options with data available: {list(candidates.keys())}"
            )
        chosen_name = mode

    chosen_df = candidates[chosen_name]
    print(f"DATA_SOURCE_MODE='{mode}' -> selected '{chosen_name}' with "
          f"{len(chosen_df)} unique images (after de-duplication).")
    print(pd.crosstab(chosen_df["top_level"], chosen_df["disease"], margins=True))
    print()

    out = chosen_df[["filepath", "disease"]].copy()
    out["source"] = chosen_df["top_level"]
    return out.reset_index(drop=True)


# ============================================================================
# 2. SEVERITY COMPUTATION VIA IMAGE SEGMENTATION (Sec 3.3)
# ============================================================================
def compute_infected_percentage(img_bgr):
    """5-stage segmentation: grayscale -> threshold -> edge detection ->
    masking -> histogram. Returns infected_area% = (P_yellow/P_total)*100
    (Eq. 6)."""
    img_bgr = cv2.resize(img_bgr, (SEG_SIZE, SEG_SIZE))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 50, 150)
    leaf_mask = cv2.bitwise_or(leaf_mask, cv2.dilate(edges, None, iterations=2))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    total_leaf_px = cv2.countNonZero(leaf_mask)
    if total_leaf_px == 0:
        return 0.0

    leaf_only = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)

    hsv = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2HSV)
    lower_infected = np.array([8, 40, 40])
    upper_infected = np.array([35, 255, 255])
    infected_mask = cv2.inRange(hsv, lower_infected, upper_infected)
    infected_mask = cv2.bitwise_and(infected_mask, infected_mask, mask=leaf_mask)

    infected_px = cv2.countNonZero(infected_mask)
    return (infected_px / total_leaf_px) * 100.0


def severity_from_pct(pct):
    """mild <25% | moderate 26-50% | severe 51-75% | profound >75% (Sec 3.3)"""
    if pct < 25:
        return "mild"
    elif pct <= 50:
        return "moderate"
    elif pct <= 75:
        return "severe"
    else:
        return "profound"


def annotate_severity(df):
    severities, pcts = [], []
    for fp in tqdm(df["filepath"], desc="Computing severity"):
        img = cv2.imread(fp)
        if img is None:
            severities.append(None)
            pcts.append(None)
            continue
        pct = compute_infected_percentage(img)
        pcts.append(pct)
        severities.append(severity_from_pct(pct))
    out = df.copy()
    out["infected_pct"] = pcts
    out["severity"] = severities
    return out.dropna(subset=["severity"]).reset_index(drop=True)


# ============================================================================
# 3. PRE-PROCESSING: standardization, normalization, rescaling (Sec 3.1.3)
# ============================================================================
def load_and_preprocess(filepath, size=IMG_SIZE):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    mean, std = img.mean(), img.std() + 1e-7
    img = (img - mean) / std
    img = (img - img.min()) / (img.max() - img.min() + 1e-7)
    return img.astype("float32")


# ============================================================================
# 4. GAN-BASED DATA AUGMENTATION (Sec 3.2, Eq. 1-5)
# ============================================================================
def build_generator(latent_dim=GAN_LATENT_DIM, img_size=IMG_SIZE):
    base = img_size // 4
    model = models.Sequential(name="generator")
    model.add(layers.Dense(base * base * 128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((base, base, 128)))
    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(3, 5, padding="same", activation="sigmoid"))
    return model


def build_discriminator(img_size=IMG_SIZE):
    model = models.Sequential(name="discriminator")
    model.add(layers.Conv2D(64, 4, strides=2, padding="same",
                             input_shape=(img_size, img_size, 3)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def train_gan_for_class(real_images, epochs=GAN_EPOCHS, batch_size=32, verbose=False):
    """One DCGAN per disease class (Eq. 1-5)."""
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=optimizers.Adam(2e-4, beta_1=0.5),
                           loss="binary_crossentropy", metrics=["accuracy"])
    discriminator.trainable = False

    gan_input = layers.Input(shape=(GAN_LATENT_DIM,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output, name="gan")
    gan.compile(optimizer=optimizers.Adam(2e-4, beta_1=0.5), loss="binary_crossentropy")

    n = real_images.shape[0]
    batch_size = max(1, min(batch_size, n))
    for epoch in range(epochs):
        idx = np.random.randint(0, n, batch_size)
        real_batch = real_images[idx]
        noise = np.random.normal(0, 1, (batch_size, GAN_LATENT_DIM))
        fake_batch = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))

        noise = np.random.normal(0, 1, (batch_size, GAN_LATENT_DIM))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if verbose and epoch % 25 == 0:
            d_loss = (np.add(d_loss_real[0], d_loss_fake[0])) / 2
            print(f"    GAN epoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}")

    return generator


def generate_synthetic_images(generator, n_images):
    noise = np.random.normal(0, 1, (n_images, GAN_LATENT_DIM))
    return generator.predict(noise, verbose=0)


def augment_dataset_with_gan(df, save_to_disk=SAVE_GAN_IMAGES_TO_DISK, output_dir=GAN_OUTPUT_DIR):
    """Trains one GAN per class and generates synthetic images using the
    paper's per-class percentages (Sec 3.2). Saves generated images to
    disk per class (Sec 3.6, step 7) when save_to_disk is True."""
    synth_images, synth_disease, synth_severity = [], [], []

    if save_to_disk:
        os.makedirs(output_dir, exist_ok=True)

    for cls in DISEASE_CLASSES:
        cls_df = df[df["disease"] == cls]
        if len(cls_df) == 0:
            continue
        print(f"Training GAN for class: {cls} ({len(cls_df)} real images)")
        real_imgs = np.stack([load_and_preprocess(fp) for fp in cls_df["filepath"]])
        generator = train_gan_for_class(real_imgs, epochs=GAN_EPOCHS,
                                         batch_size=min(32, len(real_imgs)))

        n_new = int(len(cls_df) * AUG_MULTIPLIER.get(cls, 1.0))
        fake_imgs = generate_synthetic_images(generator, n_new)

        if save_to_disk:
            class_dir = os.path.join(output_dir, cls)
            os.makedirs(class_dir, exist_ok=True)

        for i, img in enumerate(fake_imgs):
            img_uint8 = (np.clip(img, 0, 1) * 255).astype("uint8")
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            pct = compute_infected_percentage(img_bgr)
            synth_images.append(img.astype("float32"))
            synth_disease.append(cls)
            synth_severity.append(severity_from_pct(pct))

            if save_to_disk:
                out_path = os.path.join(class_dir, f"gan_{i:05d}.jpg")
                cv2.imwrite(out_path, img_bgr)

        if save_to_disk:
            print(f"  Saved {n_new} GAN-generated {cls} images to {class_dir}")

    return synth_images, synth_disease, synth_severity


# ============================================================================
# 4b. TRADITIONAL (NON-GAN) AUGMENTATION (Sec 3.1.3, ImageDataGenerator)
# ============================================================================
def build_traditional_augmenter():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="reflect",
    )


def augment_dataset_traditionally(images, disease_labels, per_image=TRADITIONAL_AUG_PER_IMAGE):
    """Generates rotation/shift/zoom/flip/brightness variants of each image
    in `images`. Disabled by default (USE_TRADITIONAL_AUGMENTATION)."""
    augmenter = build_traditional_augmenter()
    aug_images, aug_disease = [], []

    for img, disease in tqdm(zip(images, disease_labels), total=len(images),
                              desc="Traditional augmentation"):
        img_batch = np.expand_dims(img, axis=0)
        gen = augmenter.flow(img_batch, batch_size=1)
        for _ in range(per_image):
            aug_img = np.clip(next(gen)[0], 0, 1).astype("float32")
            aug_images.append(aug_img)
            aug_disease.append(disease)

    return aug_images, aug_disease


# ============================================================================
# 5. MODEL: CNN FEATURE EXTRACTOR + SVM-STYLE HEAD (Sec 3.4, Table 4)
# ============================================================================
def to_signed_onehot(y, n_classes):
    """One-hot in {-1,+1}, required by the squared-hinge/SVM output layer."""
    onehot = tf.keras.utils.to_categorical(y, n_classes)
    return 2 * onehot - 1


def build_cnn_svm_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         n_disease=len(DISEASE_CLASSES),
                         svm_c=0.01):
    """CNN-SVM hybrid for disease-type classification (3 classes), matching
    Table 4's architecture and parameter counts exactly."""
    inputs = layers.Input(shape=input_shape)

    # conv2d_12: 5x5, 96 filters, stride 2 -> 32x32x96
    x = layers.Conv2D(96, (5, 5), strides=2, padding="same", activation="relu",
                       name="conv2d_C1")(inputs)
    # maxpool_P1: 2x2, stride 2 -> 16x16x96
    x = layers.MaxPooling2D((2, 2), strides=2, padding="same", name="maxpool_P1")(x)

    # conv2d_13: 3x3, 64 filters, stride 1 -> 16x16x64 (see module docstring
    # for why stride=1 is used here instead of Table 4's listed stride=2)
    x = layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu",
                       name="conv2d_C2")(x)
    # maxpool_P2: 2x2, stride 2 -> 8x8x64
    x = layers.MaxPooling2D((2, 2), strides=2, padding="same", name="maxpool_P2")(x)

    x = layers.Flatten(name="flatten")(x)                          # 4,096
    x = layers.Dense(288, activation="relu", name="dense_fc")(x)   # 1,179,936 params

    disease_out = layers.Dense(
        n_disease, activation="linear",
        kernel_regularizer=regularizers.l2(svm_c), name="disease_output"
    )(x)

    return models.Model(inputs=inputs, outputs=disease_out, name="CNN_SVM_hybrid")


# ============================================================================
# 6. PLOTTING HELPERS
# ============================================================================
def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Disease-type accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Disease-type loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================
def main():
    # --- Step 1: collect data (Sec 3.1) ---
    print("Scanning dataset directories ...")
    df = build_dataframe(DATA_ROOTS)
    print(f"Final collected dataset: {len(df)} images total.")

    # --- Step 2: severity annotation via segmentation (Sec 3.3) ---
    print("\nAnnotating severity levels via segmentation ...")
    df = annotate_severity(df)
    print(df["severity"].value_counts())

    # --- Step 3: load real images (Sec 3.1.3) ---
    print("\nLoading real images into memory ...")
    X_real = np.stack([load_and_preprocess(fp) for fp in tqdm(df["filepath"])])
    disease_real = df["disease"].tolist()
    severity_real = df["severity"].tolist()

    # --- Step 4: GAN augmentation on the whole dataset, then split (Sec 3.2) ---
    print("\nRunning GAN augmentation on the full dataset (this can take a while) ...")
    synth_images, synth_disease, synth_severity = augment_dataset_with_gan(df)

    if synth_images:
        X_augmented = np.concatenate([X_real, np.stack(synth_images)], axis=0)
        disease_augmented = disease_real + synth_disease
        severity_augmented = severity_real + synth_severity
    else:
        X_augmented = X_real
        disease_augmented = disease_real
        severity_augmented = severity_real
    print(f"Dataset size after GAN augmentation: {len(X_augmented)} "
          f"(originally {len(X_real)} real)")

    # --- Step 5: train/val/test split, 80:20 then 80:20 (Sec 3.2/3.5) ---
    disease_to_idx = {c: i for i, c in enumerate(DISEASE_CLASSES)}
    y_augmented = np.array([disease_to_idx[d] for d in disease_augmented])
    severity_augmented_arr = np.array(severity_augmented)

    X_trainval, X_test, y_trainval, y_test, sev_trainval, sev_test = train_test_split(
        X_augmented, y_augmented, severity_augmented_arr, test_size=0.20,
        random_state=SEED, stratify=y_augmented
    )
    X_train, X_val, y_train, y_val, sev_train, sev_val = train_test_split(
        X_trainval, y_trainval, sev_trainval, test_size=0.20,
        random_state=SEED, stratify=y_trainval
    )
    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print("Class balance in training set:",
          {DISEASE_CLASSES[i]: int(c) for i, c in zip(*np.unique(y_train, return_counts=True))})

    Yd_val = to_signed_onehot(y_val, len(DISEASE_CLASSES))
    Yd_test = to_signed_onehot(y_test, len(DISEASE_CLASSES))

    # --- Step 6: traditional augmentation, if enabled (Sec 3.1.3) ---
    if USE_TRADITIONAL_AUGMENTATION:
        print(f"\nRunning traditional augmentation ({TRADITIONAL_AUG_PER_IMAGE}x "
              f"per training image) ...")
        trad_images, trad_disease_idx = augment_dataset_traditionally(
            X_train, [DISEASE_CLASSES[i] for i in y_train]
        )
        X_train = np.concatenate([X_train, np.stack(trad_images)], axis=0)
        y_train = np.concatenate([y_train,
                                   np.array([disease_to_idx[d] for d in trad_disease_idx])])
        print(f"Training set size after traditional augmentation: {len(X_train)}")

    Yd_train = to_signed_onehot(y_train, len(DISEASE_CLASSES))

    print(f"\nFinal training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")

    # --- Step 7: build + compile model (Sec 3.4, Table 4) ---
    model = build_cnn_svm_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="squared_hinge",
        metrics=["accuracy"],
    )
    model.summary()

    # --- Step 8: train (Sec 3.5.1) ---
    history = model.fit(
        X_train, Yd_train,
        validation_data=(X_val, Yd_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # --- Step 9: evaluate on held-out test set (Sec 4) ---
    pred_disease_raw = model.predict(X_test)
    pred_disease = np.argmax(pred_disease_raw, axis=1)
    true_disease = np.argmax(Yd_test, axis=1)

    print("\n=== Disease-type classification ===")
    print(f"Accuracy: {accuracy_score(true_disease, pred_disease) * 100:.2f}%")
    print(classification_report(true_disease, pred_disease, target_names=DISEASE_CLASSES))
    cm_disease = confusion_matrix(true_disease, pred_disease)
    print("Confusion matrix (disease type):")
    print(cm_disease)

    print("\n=== Severity distribution within the test set (rule-based, Sec 3.3) ===")
    print(pd.Series(sev_test).value_counts())

    # --- Step 10: plots (Fig. 9 & 10 style) ---
    plot_confusion_matrix(cm_disease, DISEASE_CLASSES, "Confusion matrix — disease type")
    plot_training_curves(history)

    # --- Step 11: save model ---
    model.save("paddy_cnn_svm_hybrid.keras")
    print("\nSaved model to paddy_cnn_svm_hybrid.keras")

    return model, history, df


if __name__ == "__main__":
    main()
