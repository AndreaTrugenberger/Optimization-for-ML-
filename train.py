#Trains CNNs on MNIST and CIFAR-10 with Adam, SignSGD, Lion, and AdaHessian optimizers.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json, os, argparse, time
from optimizers import SignSGD, Lion, AdaHessian

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠  CUDA not available, using CPU.")
    return device


# CNN Models for MNIST and CIFAR-10

class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset):
    return CNN_MNIST() if dataset.lower() == "mnist" else CNN_CIFAR()


# Data loading

NUM_WORKERS = 2

def get_loaders(dataset, batch_size, data_dir="./data"):
    dataset = dataset.lower()
    pin = torch.cuda.is_available()   # pin_memory speeds up CPU→GPU transfer

    if dataset == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tfm)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    elif dataset == "cifar10":
        tfm_tr = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        tfm_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tfm_tr)
        test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm_te)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
        drop_last=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
        persistent_workers=True,
    )
    return train_loader, test_loader


# Evaluation and training loop

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * len(y)
        correct    += out.argmax(1).eq(y).sum().item()
        n          += len(y)
    model.train()
    return total_loss / n, correct / n


# Optimizers

DEFAULT_LRS = {
    "mnist":   {"adam": 1e-3, "signsgd": 1e-3, "lion": 1e-4, "adahessian": 0.1},
    "cifar10": {"adam": 5e-4, "signsgd": 5e-4, "lion": 5e-5, "adahessian": 0.05},
}

def build_optimizer(name, params, lr, weight_decay=1e-4):
    name = name.lower()
    if   name == "adam":        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "signsgd":     return SignSGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "lion":        return Lion(params, lr=lr, weight_decay=weight_decay)
    elif name == "adahessian":  return AdaHessian(params, lr=lr, weight_decay=weight_decay)
    else: raise ValueError(f"Unknown optimizer: {name}")


# Training for one run
def train_one_run(opt_name, lr, epochs, batch_size, seed, device, data_dir, dataset):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader, test_loader = get_loaders(dataset, batch_size, data_dir)
    model     = build_model(dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(opt_name, model.parameters(), lr)
    is_ada    = opt_name.lower() == "adahessian"

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "epoch_time": [],
        "opt": opt_name, "lr": lr, "seed": seed,
        "batch_size": batch_size, "dataset": dataset, "arch": "CNN",
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss, correct, n = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

            out  = model(x)
            loss = criterion(out, y)
            loss.backward(create_graph=is_ada)
            optimizer.step()

            running_loss += loss.item() * len(y)
            correct      += out.argmax(1).eq(y).sum().item()
            n            += len(y)

        tr_loss, tr_acc = running_loss / n, correct / n
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["epoch_time"].append(elapsed)

        print(f"[CNN | {dataset.upper():7s} | {opt_name:12s} | "
              f"bs={batch_size:4d} | seed={seed}] "
              f"Ep {epoch:2d}/{epochs}  "
              f"tr={tr_acc:.4f}  te={te_acc:.4f}  ({elapsed:.1f}s)")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers",  nargs="+",
                        default=["adam", "signsgd", "lion", "adahessian"])
    parser.add_argument("--datasets",    nargs="+", default=["mnist"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--seeds",       nargs="+",  type=int, default=[0, 1, 2])
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--out_dir",     default="./results")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    for dataset in args.datasets:
        lrs = DEFAULT_LRS.get(dataset.lower(), DEFAULT_LRS["mnist"])
        for batch_size in args.batch_sizes:
            for opt in args.optimizers:
                lr = args.lr if args.lr is not None else lrs.get(opt.lower(), 1e-3)
                for seed in args.seeds:
                    hist = train_one_run(opt, lr, args.epochs, batch_size,
                                        seed, device, args.data_dir, dataset)
                    all_results.append(hist)

    out_path = os.path.join(args.out_dir, "results.json")
    # Append to existing results if file exists
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        all_results = existing + all_results
        print(f"  (Appended to existing {len(existing)} runs)")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Saved {len(all_results)} total runs to {out_path}")
    print("Now run: python3 plot_results.py")


if __name__ == "__main__":
    main()
