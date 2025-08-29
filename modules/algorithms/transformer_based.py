import logging
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODELS_LOCATION = "./models/transformer"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def upsample_to_match(df, label_col):
    grouped = df.groupby(label_col)
    target_size = grouped.size().max()
    balanced_groups = [
        resample(group, replace=True, n_samples=target_size, random_state=42)
        for _, group in grouped
    ]
    return (
        pd.concat(balanced_groups)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def prepare_data(df):
    sentiment_map = {-1: 0, 0: 1, 1: 2}
    df["appearance_label"] = df["appearance_sentiment"].map(sentiment_map)
    df["palate_label"] = df["palate_sentiment"].map(sentiment_map)

    df["appearance_label"] = df["appearance_label"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    df["palate_label"] = df["palate_label"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    df["processed_text"] = df["processed_text"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )

    df = df.dropna(subset=["appearance_label", "palate_label"])

    df["joint_label"] = (
        df["appearance_label"].astype(str) + "_" + df["palate_label"].astype(str)
    )
    df = upsample_to_match(df, "joint_label")

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_df = None
    test_df = None

    for train_idx, test_idx in strat_split.split(df, df["joint_label"]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    final_train_df = None
    val_df = None

    for train_idx, val_idx in val_split.split(train_df, train_df["joint_label"]):
        final_train_df = train_df.iloc[train_idx].copy()
        val_df = train_df.iloc[val_idx].copy()

    return final_train_df, val_df, test_df


class BeerDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.texts = dataframe["processed_text"].tolist()
        self.appearance_labels = dataframe["appearance_label"].tolist()
        self.palate_labels = dataframe["palate_label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "appearance_label": torch.tensor(
                self.appearance_labels[idx], dtype=torch.long
            ),
            "palate_label": torch.tensor(self.palate_labels[idx], dtype=torch.long),
        }


class MultiAspectModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.shared_dropout = nn.Dropout(0.3)

        self.appearance_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

        self.palate_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.shared_dropout(outputs.last_hidden_state[:, 0, :])
        return self.appearance_head(cls_output), self.palate_head(cls_output)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = f.log_softmax(logits, dim=1)
            ce_loss = -(true_dist * log_probs).sum(dim=1)
        else:
            ce_loss = f.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss *= alpha_t
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


def compute_class_weights(labels):
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, device, loss_fn_app, loss_fn_pal, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        appearance_label = batch["appearance_label"].to(device)
        palate_label = batch["palate_label"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            app_logits, pal_logits = model(input_ids, attention_mask)
            loss1 = loss_fn_app(app_logits, appearance_label)
            loss2 = loss_fn_pal(pal_logits, palate_label)
            loss = 0.6 * loss1 + 0.4 * loss2

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def compute_val_loss(model, loader, device, loss_fn_app, loss_fn_pal):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance_label = batch["appearance_label"].to(device)
            palate_label = batch["palate_label"].to(device)

            with autocast(device_type="cuda"):
                app_logits, pal_logits = model(input_ids, attention_mask)
                loss1 = loss_fn_app(app_logits.float(), appearance_label)
                loss2 = loss_fn_pal(pal_logits.float(), palate_label)
                loss = 0.6 * loss1 + 0.4 * loss2

            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, split="Validation"):
    model.eval()
    all_preds_app, all_labels_app = [], []
    all_preds_pal, all_labels_pal = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{split} Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            appearance_label = batch["appearance_label"].to(device)
            palate_label = batch["palate_label"].to(device)

            with autocast(device_type="cuda"):
                app_logits, pal_logits = model(input_ids, attention_mask)

            all_preds_app.extend(torch.argmax(app_logits, dim=1).cpu().numpy())
            all_labels_app.extend(appearance_label.cpu().numpy())
            all_preds_pal.extend(torch.argmax(pal_logits, dim=1).cpu().numpy())
            all_labels_pal.extend(palate_label.cpu().numpy())

    f1_app = f1_score(all_labels_app, all_preds_app, average="macro")
    f1_pal = f1_score(all_labels_pal, all_preds_pal, average="macro")
    avg_f1 = (f1_app + f1_pal) / 2

    LOGGER.info(
        f"{split} - Appearance F1: {f1_app:.4f}, Palate F1: {f1_pal:.4f}, Avg F1: {avg_f1:.4f}"
    )
    LOGGER.info(f"{split} - Appearance preds: {Counter(all_preds_app)}")
    LOGGER.info(f"{split} - Palate preds: {Counter(all_preds_pal)}")
    LOGGER.info(
        "Appearance Classification Report:\n"
        + classification_report(all_labels_app, all_preds_app)
    )
    LOGGER.info(
        "Palate Classification Report:\n"
        + classification_report(all_labels_pal, all_preds_pal)
    )

    return avg_f1


def get_optimizer_grouped_parameters(model, base_lr=2e-5, lr_decay=0.95):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    if hasattr(model.bert, "encoder") and hasattr(model.bert.encoder, "layer"):
        layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
    elif hasattr(model.bert, "transformer") and hasattr(model.bert.transformer, "layer"):
        layers = [model.bert.embeddings] + list(model.bert.transformer.layer)
    else:
        raise AttributeError("Unsupported transformer model type for optimizer grouping.")

    num_layers = len(layers)

    for i, layer in enumerate(layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - i))
        params = list(layer.named_parameters())
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "lr": layer_lr,
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "lr": layer_lr,
                "weight_decay": 0.0,
            },
        ]

    optimizer_grouped_parameters += [
        {"params": model.appearance_head.parameters(), "lr": base_lr},
        {"params": model.palate_head.parameters(), "lr": base_lr},
    ]

    return optimizer_grouped_parameters


def run_pipeline(df, model_name="prajjwal1/bert-mini", epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df, val_df, test_df = prepare_data(df)

    LOGGER.info(
        f"Train Appearance Label Distribution: {Counter(train_df['appearance_label'])}"
    )
    LOGGER.info(f"Train Palate Label Distribution: {Counter(train_df['palate_label'])}")

    train_loader = DataLoader(
        BeerDataset(train_df, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    val_loader = DataLoader(
        BeerDataset(val_df, tokenizer),
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )
    test_loader = DataLoader(
        BeerDataset(test_df, tokenizer),
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )

    model = MultiAspectModel(model_name).to(device)

    app_weights = compute_class_weights(train_df["appearance_label"].tolist()).to(
        device
    )
    pal_weights = compute_class_weights(train_df["palate_label"].tolist()).to(device)

    loss_fn_app = FocalLoss(alpha=app_weights, gamma=2, label_smoothing=0.15)
    loss_fn_pal = FocalLoss(alpha=pal_weights, gamma=2, label_smoothing=0.15)

    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, base_lr=2e-5), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7
    )

    best_f1 = 0.0
    best_val_loss = float("inf")
    patience, wait = 2, 0

    save_dir = f"{MODELS_LOCATION}/{model_name.replace('/', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        LOGGER.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, device, loss_fn_app, loss_fn_pal, scaler
        )
        LOGGER.info(f"Training Loss: {train_loss:.4f}")

        val_f1 = evaluate(model, val_loader, device, split="Validation")
        val_loss = compute_val_loss(model, val_loader, device, loss_fn_app, loss_fn_pal)
        scheduler.step(val_loss)

        if val_f1 > best_f1 or val_loss < best_val_loss:
            best_f1 = val_f1
            best_val_loss = val_loss
            wait = 0
            torch.save(
                model.state_dict(),
                f"{MODELS_LOCATION}/{model_name.replace('/', '_')}/{model_name.replace('/', '_')}.pt",
            )
            LOGGER.info(
                f"New best model saved with F1: {best_f1:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            wait += 1
            LOGGER.info(f"No improvement. Patience: {wait}/{patience}")
            if wait >= patience:
                LOGGER.info("Early stopping triggered.")
                break

    model.load_state_dict(
        torch.load(f"{MODELS_LOCATION}/{model_name.replace('/', '_')}/{model_name.replace('/', '_')}.pt")
    )
    evaluate(model, test_loader, device, split="Final Test")


def resume_transformer_training(df, model_name, checkpoint_path, start_epoch=0, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df, val_df, test_df = prepare_data(df)

    train_loader = DataLoader(
        BeerDataset(train_df, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    val_loader = DataLoader(
        BeerDataset(val_df, tokenizer),
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )
    test_loader = DataLoader(
        BeerDataset(test_df, tokenizer),
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )

    model = MultiAspectModel(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    app_weights = compute_class_weights(train_df["appearance_label"].tolist()).to(device)
    pal_weights = compute_class_weights(train_df["palate_label"].tolist()).to(device)

    loss_fn_app = FocalLoss(alpha=app_weights, gamma=2, label_smoothing=0.15)
    loss_fn_pal = FocalLoss(alpha=pal_weights, gamma=2, label_smoothing=0.15)

    optimizer = optim.AdamW(
        get_optimizer_grouped_parameters(model, base_lr=2e-5), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7
    )

    best_f1 = 0.0
    best_val_loss = float("inf")
    patience, wait = 2, 0

    save_dir = f"{MODELS_LOCATION}/{model_name.replace('/', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        LOGGER.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, device, loss_fn_app, loss_fn_pal, scaler
        )
        LOGGER.info(f"Training Loss: {train_loss:.4f}")

        val_f1 = evaluate(model, val_loader, device, split="Validation")
        val_loss = compute_val_loss(model, val_loader, device, loss_fn_app, loss_fn_pal)
        scheduler.step(val_loss)

        if val_f1 > best_f1 or val_loss < best_val_loss:
            best_f1 = val_f1
            best_val_loss = val_loss
            wait = 0
            torch.save(
                model.state_dict(),
                f"{save_dir}/{model_name.replace('/', '_')}.pt",
            )
            LOGGER.info(
                f"New best model saved with F1: {best_f1:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            wait += 1
            LOGGER.info(f"No improvement. Patience: {wait}/{patience}")

    model.load_state_dict(
        torch.load(f"{MODELS_LOCATION}/{model_name.replace('/', '_')}/{model_name.replace('/', '_')}.pt")
    )
    evaluate(model, test_loader, device, split="Final Test")

def transformer_based_trainer(model_option: int = 1):
    from modules.utils.utilities import load_dataframe_from_database
    model = "prajjwal1/bert-mini" if model_option == 1 else "distilbert-base-uncased"
    set_seed()
    df = load_dataframe_from_database(is_target=True)
    run_pipeline(df, model_name=model)

def resume_training():
    from modules.utils.utilities import load_dataframe_from_database
    model = "distilbert-base-uncased"
    set_seed()
    df = load_dataframe_from_database(is_target=True)
    resume_transformer_training(df, model_name=model, checkpoint_path="./models/transformer/distilbert-base-uncased/distilbert-base-uncased.pt", start_epoch=6, epochs=10)