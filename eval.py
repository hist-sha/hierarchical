import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch

from collections import defaultdict
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision import transforms
from tqdm import tqdm


def extract_coordinates(filename):
    filename = os.path.basename(filename)
    match = re.match(r"(\d+)_(\d+)\.jpeg", filename)
    return (int(match.group(1)), int(match.group(2))) if match else None


def extract_features(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()


def extract_all_features(model, patch_paths, device):
    features = []
    coords = []
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for p in tqdm(patch_paths, desc="Extracting features"):
        features.append(extract_features(model, p, transform, device))
        coord = extract_coordinates(p)
        coords.append(coord)
    return np.array(features), coords


def cluster_features(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def train_model(model, train_loader, criterion, optimizer, num_epochs=5, device="gpu"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = batch["features"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds.double() / total_preds
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
        )

    print("Training complete.")


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch["features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"acc: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")
    print("conf mtrx:")
    print(cm)


def get_predictions_and_coords_from_loader(model, loader, device):
    model.eval()
    predictions = []
    coords = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["features"].to(device)
            batch_coords = batch["coords"]
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            coords.extend(batch_coords)
    return predictions, coords


def plot_patches_with_labels(coords, labels, patch_size=224):
    class_colors = {0: "red", 1: "green", 2: "blue", 3: "orange", 4: "purple"}

    patches_by_coord = defaultdict(list)
    for i, coord in enumerate(coords):
        x, y = coord
        label = labels[i]
        color = class_colors.get(label, "black")
        patches_by_coord[(x, y)].append((patch_size, patch_size, color))

    _, ax = plt.subplots(figsize=(10, 10))
    for coord, patches in patches_by_coord.items():
        x, y = coord
        for patch in patches:
            patch_size_x, patch_size_y, color = patch
            if patch_size_x is None or patch_size_y is None:
                print(
                    f"Warning: None size detected for patch at {coord}. Skipping this patch."
                )
                continue
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y),
                    1,
                    1,
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.5,
                )
            )

    ax.set_xlim(0, 330)
    ax.set_ylim(0, 250)
    ax.set_aspect("equal")
    ax.set_title("Patch positions with predicted labels")

    legend_handles = [
        mpatches.Patch(color=color, label=f"class {i}")
        for i, color in class_colors.items()
    ]
    ax.legend(handles=legend_handles)

    plt.gca().invert_yaxis()
    plt.show()
