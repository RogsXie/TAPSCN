import argparse
import math
import os
import random
import time

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from model.model import TAPSCN
from utils.dataprocess import load_data, nor_pca, traintwo_patch
from utils.evulate import calculate_metrics
from utils.generatepic import all_create_colored_classified_image, plot_category_colors
from utils.output import save_metrics_and_accuracies

parser = argparse.ArgumentParser("TAPSCN")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test_freq", type=int, default=1)
parser.add_argument("--epoches", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument(
    "--dataset",
    default="Houston",
    choices=["Muufl", "Trento", "Houston", "Houston18", "Augsburg"],
)
parser.add_argument("--num_classes", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--train_num", type=int, default=50)
parser.add_argument("--patches1", type=int, default=11)
parser.add_argument("--more0", action="store_true", default=False)
parser.add_argument("--eval_chunk", type=int, default=5000)
parser.add_argument(
    "--proto_refresh",
    type=int,
    default=10,
    help="Re-compute class prototypes every N epochs",
)
parser.add_argument(
    "--lambda_re",
    type=float,
    default=0.01,
    help="Weight for MSE reconstruction loss: 0.01",
)
parser.add_argument(
    "--lambda_c", type=float, default=1, help="Weight for CLIP contrastive loss: 1"
)
args = parser.parse_args()
print(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
os.environ["PYTHONHASHSEED"] = str(args.seed)


def generate_predicted_map(model, patches1):
    dataset_name = args.dataset
    best_model_pth = os.path.join(dataset_name, "best_model.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(best_model_pth, weights_only=True))
    model.eval().to(device)
    Data1, Data2, gt, _, _ = load_data(dataset_name)
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    hsi_data, dsm_data, _NC = nor_pca(Data1, Data2, ispca=True)
    if dsm_data.ndim == 2:
        dsm_data = np.expand_dims(dsm_data, axis=-1)
    lidar_seg_gpu = torch.from_numpy(dsm_data).to(device)
    patchsize = patches1
    half_patch = patchsize // 2
    hsi_padded = np.pad(
        hsi_data,
        ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
        mode="edge",
    )
    dsm_padded = np.pad(
        dsm_data,
        ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
        mode="edge",
    )
    height, width = hsi_data.shape[:2]
    predicted_map = np.zeros((height, width), dtype=int)
    coordinates = [(i, j) for i in range(height) for j in range(width)]
    for idx in range(0, len(coordinates), args.batch_size):
        batch_coords = coordinates[idx : idx + args.batch_size]
        hsi_batch, dsm_batch = ([], [])
        for i, j in batch_coords:
            hsi_patch = hsi_padded[i : i + patchsize, j : j + patchsize, :]
            dsm_patch = dsm_padded[i : i + patchsize, j : j + patchsize, :]
            hsi_batch.append(torch.from_numpy(hsi_patch).permute(2, 0, 1).float())
            dsm_batch.append(torch.from_numpy(dsm_patch).permute(2, 0, 1).float())
        with torch.no_grad():
            hsi_stack = torch.stack(hsi_batch).to(device)
            dsm_stack = torch.stack(dsm_batch).to(device)
            logits, _, _ = model(hsi_stack, dsm_stack, lidar_seg_gpu)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        for (i, j), pred in zip(batch_coords, preds):
            predicted_map[i, j] = pred + 1
    return predicted_map


LABEL_DESCRIPTIONS = {
    "Houston": [
        ["This is healthy grass area."],
        ["This is stressed grass area."],
        ["This is synthetic grass area."],
        ["This is trees area."],
        ["This is soil area."],
        ["This is water area."],
        ["This is residential area."],
        ["This is commercial area."],
        ["This is road area."],
        ["This is highway area."],
        ["This is railway area."],
        ["This is parking lot area."],
        ["This is another type of parking lot area."],
        ["This is tennis court area."],
        ["This is running track area."],
    ],
    "Trento": [
        ["This is apple orchard area."],
        ["This is buildings area."],
        ["This is ground surface area."],
        ["This is woodland area."],
        ["This is vineyard area."],
        ["This is road area."],
    ],
    "Muufl": [
        ["This is trees area."],
        ["This is mostly grass area."],
        ["This is mixed ground surface area."],
        ["This is dirt and sand area."],
        ["This is road area."],
        ["This is water area."],
        ["This is buildings area."],
        ["This is shadow area."],
        ["This is sidewalk area."],
        ["This is yellow curb area."],
        ["This is cloth panels area."],
    ],
    "Augsburg": [
        ["This is forest area."],
        ["This is residential area."],
        ["This is industrial area."],
        ["This is low plants area."],
        ["This is allotment area."],
        ["This is commercial area."],
        ["This is water area."],
    ],
    "Houston18": [
        ["This is healthy grass area."],
        ["This is stressed grass area."],
        ["This is artificial turf area."],
        ["This is evergreen trees area."],
        ["This is deciduous trees area."],
        ["This is bare earth area."],
        ["This is water area."],
        ["This is residential buildings area."],
        ["This is non-residential buildings area."],
        ["This is roads area."],
        ["This is sidewalks area."],
        ["This is crosswalks area."],
        ["This is major thoroughfares area."],
        ["This is highways area."],
        ["This is railways area."],
        ["This is paved parking lots area."],
        ["This is unpaved parking lots area."],
        ["This is cars area."],
        ["This is trains area."],
        ["This is stadium seats area."],
    ],
}


def build_label_tokenize(dataset: str, device: torch.device) -> list[list]:
    return [
        [clip.tokenize(prompt).to(device) for prompt in prompts]
        for prompts in LABEL_DESCRIPTIONS[dataset]
    ]


@torch.no_grad()
def compute_class_prototypes(
    model: nn.Module, label_tokenize: list[list], device: torch.device
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prototypes = []
    for class_tokens in label_tokenize:
        tokens = torch.cat(class_tokens, dim=0).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=1)
        proto = feats.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        prototypes.append(proto)
    if was_training:
        model.train()
    return torch.stack(prototypes, dim=0)


@torch.no_grad()
def compute_core_class_prototypes(
    model: nn.Module, label_tokenize: list[list], device: torch.device
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    prototypes = []
    for class_tokens in label_tokenize:
        tokens = class_tokens[0].to(device)
        feat = model.encode_text(tokens)
        feat = F.normalize(feat, dim=1)
        proto = feat.squeeze(0)
        prototypes.append(proto)
    if was_training:
        model.train()
    return torch.stack(prototypes, dim=0)


def create_dataloader():
    Data1, Data2, gt, train_labels, test_labels = load_data(args.dataset)
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    Data1, Data2, _NC = nor_pca(Data1, Data2, ispca=True)
    pad_width = args.patches1 // 2
    TrainPatch1, TrainPatch2, TrainLabel = traintwo_patch(
        Data1, Data2, args.patches1, pad_width, train_labels, args.more0
    )
    TestPatch1, TestPatch2, TestLabel = traintwo_patch(
        Data1, Data2, args.patches1, pad_width, test_labels, args.more0
    )
    train_dataset = Data.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    print(f"trainset num : {len(train_dataset)}")
    print(f"testset num  : {len(TestLabel)}")
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    return (train_loader, (TestPatch1, TestPatch2, TestLabel), Data2)


def fast_evaluate(model, test_tensors, device, lidar_seg_gpu, chunk=5000):
    TestPatch1, TestPatch2, TestLabel = test_tensors
    criterion = nn.CrossEntropyLoss()
    model.eval()
    n, all_preds, running_loss = (len(TestLabel), [], 0.0)
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            hsi = TestPatch1[s:e].to(device)
            lidar = TestPatch2[s:e].to(device)
            labels = TestLabel[s:e].to(device)
            logits, _, _ = model(hsi, lidar, lidar_seg_gpu)
            running_loss += criterion(logits, labels).item() * (e - s)
            all_preds.append(torch.max(logits, 1)[1].cpu())
    all_preds = torch.cat(all_preds)
    return (
        running_loss / n,
        100.0 * (all_preds == TestLabel).sum().item() / n,
        all_preds,
    )


def train(
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    test_tensors,
    device,
    lidar_seg_gpu,
    label_tokenize,
):
    best_val_acc = float("-inf")
    best_tra_acc = float("inf")
    best_model_path = os.path.join(args.dataset, "best_model.pth")
    class_prototypes = compute_class_prototypes(model, label_tokenize, device)
    for epoch in range(args.epoches):
        if epoch > 0 and epoch % args.proto_refresh == 0:
            class_prototypes = compute_class_prototypes(model, label_tokenize, device)
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for hsi, lidar, labels in train_loader:
            max_label = labels.max().item()
            hsi, lidar, labels = (hsi.to(device), lidar.to(device), labels.to(device))
            optimizer.zero_grad()
            text = torch.cat([random.choice(label_tokenize[int(k)]) for k in labels])
            text_proto = class_prototypes[labels.long()]
            logits, re_loss, loss_c = model(
                hsi, lidar, lidar_seg_gpu, text=text, text_proto=text_proto
            )
            loss = criterion(logits, labels) + re_loss + loss_c
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.max(logits, 1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        avg_loss = total_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        if (epoch + 1) % args.test_freq == 0:
            val_loss, val_acc, _ = fast_evaluate(
                model, test_tensors, device, lidar_seg_gpu, args.eval_chunk
            )
            print(
                f"Epoch [{epoch + 1:03d}/{args.epoches}] | Train Loss {avg_loss:.4f}  Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}  Acc {val_acc:.2f}%"
            )
            if val_acc > best_val_acc:
                os.makedirs(args.dataset, exist_ok=True)
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"  ↓ Best model saved ({best_val_acc:.2f}%)")
        scheduler.step()
    return best_model_path


def test(model, test_tensors, best_model_path, num_classes, device, lidar_seg_gpu):
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    _, _, all_preds = fast_evaluate(
        model, test_tensors, device, lidar_seg_gpu, args.eval_chunk
    )
    TestLabel = test_tensors[2]
    n = len(TestLabel)
    correct = (all_preds == TestLabel).sum().item()
    cm = np.zeros((num_classes, num_classes))
    for t, p in zip(TestLabel.numpy(), all_preds.numpy()):
        cm[t, p] += 1
    OA, AA, Kappa, class_acc = calculate_metrics(cm, n, correct)
    print(f"\nOA: {OA:.2f}%  AA: {AA:.2f}%  Kappa: {Kappa:.4f}")
    for i, a in enumerate([x * 100 for x in class_acc]):
        print(f"  {a:.2f}")
    print(f" {OA:.2f}")
    print(f" {AA:.2f}")
    print(f" {Kappa:.2f}")
    return (cm, class_acc, OA, AA, Kappa)


def extract_features_for_tsne(
    model, test_tensors, device, lidar_seg_gpu, chunk: int = 5000
):
    TestPatch1, TestPatch2, TestLabel = test_tensors
    model.eval()
    all_feats, all_labels = ([], [])
    n = len(TestLabel)
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            hsi = TestPatch1[s:e].to(device)
            lidar = TestPatch2[s:e].to(device)
            logits, _, _ = model(hsi, lidar, lidar_seg_gpu)
            all_feats.append(logits.cpu().numpy())
            all_labels.append(TestLabel[s:e].numpy())
    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return (features, labels)


def plot_tsne(
    model,
    test_tensors,
    device,
    lidar_seg_gpu,
    dataset: str,
    num_classes: int,
    max_samples: int = 5000,
    perplexity: int = 50,
    save_path: str = None,
):
    import matplotlib
    from sklearn.manifold import TSNE

    matplotlib.use("Agg")
    import matplotlib.cm as cm_lib
    import matplotlib.pyplot as plt

    print("\n[t-SNE] Extracting features …")
    features, labels = extract_features_for_tsne(
        model, test_tensors, device, lidar_seg_gpu
    )
    n_total = len(labels)
    if n_total > max_samples:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n_total, size=max_samples, replace=False)
        features, labels = (features[idx], labels[idx])
        print(f"[t-SNE] Down-sampled to {max_samples} / {n_total} samples.")
    print(f"[t-SNE] Running TSNE (perplexity={perplexity}) on {len(labels)} samples …")
    embed = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(
        features
    )
    cmap = cm_lib.get_cmap("tab20", num_classes)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        ax.scatter(
            embed[mask, 0],
            embed[mask, 1],
            s=6,
            alpha=0.7,
            color=cmap(c),
            label=f"Class {c + 1}",
            linewidths=0,
        )
    ax.legend(markerscale=3, fontsize=7, loc="best", ncol=max(1, num_classes // 10))
    ax.axis("off")
    plt.tight_layout()
    if save_path is None:
        os.makedirs(dataset, exist_ok=True)
        save_path = os.path.join(dataset, f"tsne_{dataset}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[t-SNE] Figure saved → {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_config = {
        "Houston": (20, 1),
        "Trento": (20, 1),
        "Muufl": (20, 2),
        "Augsburg": (20, 1),
        "Houston18": (20, 1),
    }
    band1, band2 = dataset_config[args.dataset]
    train_loader, test_tensors, Data2 = create_dataloader()
    LiDAR_seg = Data2
    if LiDAR_seg.ndim == 2:
        LiDAR_seg = np.expand_dims(LiDAR_seg, axis=-1)
    lidar_seg_gpu = torch.from_numpy(LiDAR_seg.astype(np.float32)).to(device)
    label_tokenize = build_label_tokenize(args.dataset, device)
    model = TAPSCN(
        C1=band1,
        C2=band2,
        Classes=args.num_classes,
        image_size=args.patches1,
        embed_dim=128,
        context_length=77,
        vocab_size=49408,
        transformer_width=64,
        lambda_re=args.lambda_re,
        lambda_c=args.lambda_c,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epoches // 10, gamma=args.gamma
    )
    torch.cuda.synchronize()
    t0 = time.time()
    best_model_path = train(
        model,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        test_tensors,
        device,
        lidar_seg_gpu,
        label_tokenize=label_tokenize,
    )
    torch.cuda.synchronize()
    train_time = time.time() - t0
    torch.cuda.synchronize()
    t0 = time.time()
    results = test(
        model, test_tensors, best_model_path, args.num_classes, device, lidar_seg_gpu
    )
    torch.cuda.synchronize()
    test_time = time.time() - t0
    save_metrics_and_accuracies(
        results[1],
        results[2],
        results[3],
        results[4],
        train_time,
        test_time,
        args.dataset,
    )
    predicted_map = generate_predicted_map(model, args.patches1)
    all_create_colored_classified_image(predicted_map, args.dataset)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    plot_tsne(
        model,
        test_tensors,
        device,
        lidar_seg_gpu,
        dataset=args.dataset,
        num_classes=args.num_classes,
        max_samples=7000,
        perplexity=40,
    )
