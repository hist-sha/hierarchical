# dataset.py
import torch
import logging
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(self, features, labels, coords):
        self.features = features
        self.labels = labels
        self.coords = coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        coord = self.coords[idx]

        if feature is None or label is None or coord is None:
            logger.error(
                f"Error at index {idx}: one of the data points is None. Coordinates: {coord}"
            )
            return None

        return {
            "features": torch.tensor(feature, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
            "coords": coord,
        }


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        logger.warning("Empty batch after filtering out None items.")
        return None
    features = torch.stack([item["features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    coords = [item["coords"] for item in batch]
    return {"features": features, "labels": labels, "coords": coords}


def create_dataloader(features, labels, coords, batch_size=32):
    dataset = CustomDataset(features, labels, coords)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
