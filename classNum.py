import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd


class SimpleNet(nn.Module):
    """Simple feedforward network for MNIST."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_model(model: nn.Module, loader: DataLoader, lr: float, device: torch.device) -> None:
    """Train the model for one epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, limit: int = 1000) -> float:
    """Evaluate accuracy on up to ``limit`` samples."""
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if total >= limit:
                break
    return correct / min(total, limit)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    results = []
    for lr in [0.01, 0.001]:
        for hidden in [128, 256, 512]:
            for batch in [32, 64, 128]:
                train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=batch)
                model = SimpleNet(hidden)
                train_model(model, train_loader, lr, device)
                acc = evaluate(model, test_loader, device)
                results.append({"learningRate": lr, "hidden": hidden, "batchSize": batch, "accuracy": acc})
                print(f"lr={lr}, hidden={hidden}, batch={batch} -> accuracy={acc:.4f}")

    df = pd.DataFrame(results)
    print("\nSummary:\n", df)


if __name__ == "__main__":
    main()
