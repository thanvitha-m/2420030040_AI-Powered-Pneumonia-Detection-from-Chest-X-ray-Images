from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "chest_xray"

# Training preprocessing + augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/test preprocessing — no random augmentation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(batch_size=16, validation_ratio=0.2):

    # Create two versions of the same training directory:
    # one with augmentation and one without augmentation.
    train_aug_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform
    )

    train_eval_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=eval_transform
    )

    total_size = len(train_aug_dataset)
    validation_size = int(total_size * validation_ratio)
    training_size = total_size - validation_size

    # Fixed seed makes the split reproducible.
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        total_size,
        generator=generator
    ).tolist()

    train_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    train_dataset = Subset(
        train_aug_dataset,
        train_indices
    )

    validation_dataset = Subset(
        train_eval_dataset,
        validation_indices
    )

    # Keep the original test set untouched.
    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_aug_dataset.classes
    )


if __name__ == "__main__":

    train_loader, validation_loader, test_loader, classes = get_dataloaders()

    print("Dataset loaded successfully!")
    print("Classes:", classes)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(validation_loader.dataset))
    print("Test images:", len(test_loader.dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))