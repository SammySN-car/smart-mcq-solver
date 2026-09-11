import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from src.utils import set_seed, mapk, clear_vram


class CustomDataset3(Dataset):
    def __init__(self, df, model_name, options, is_test=False):
        self.df = df.reset_index(drop=True)
        self.options = options
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, attention_masks = [], []

        for option in self.options:
            text = f"{row['prompt']} {row[option]}"
            encoding = self.tokenizer(
                text, max_length=128, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids.append(encoding['input_ids'].squeeze(0))
            attention_masks.append(encoding['attention_mask'].squeeze(0))

        if self.is_test:
            return torch.stack(input_ids), torch.stack(attention_masks)

        return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(self.options.index(row['answer']), dtype=torch.long)


class MiniLMModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        b, n, l = input_ids.size()
        flat_ids = input_ids.view(b * n, l)
        flat_mask = attention_mask.view(b * n, l)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls_out = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_out)

        return logits.view(b, n)


def train_minilm(train, test, options, model_name, config, device, n_folds=3):
    seed = config.get('seed', 1)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 3)

    n_train = len(train)
    n_test = len(test)

    test_minilm_matrix = np.zeros((n_test, 5))
    train_minilm_oof = np.zeros((n_train, 5))

    skf3 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf3.split(np.arange(n_train), train['answer'].values)):
        set_seed(seed + fold)

        train_dataset = CustomDataset3(train.iloc[train_idx], model_name, options)
        val_dataset = CustomDataset3(train.iloc[val_idx], model_name, options)
        test_dataset = CustomDataset3(test, model_name, options, is_test=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model_3 = MiniLMModel(model_name).to(device)
        opt3 = torch.optim.AdamW(model_3.parameters(), lr=2e-5, weight_decay=0.01)
        criterion_3 = nn.CrossEntropyLoss()

        total_steps = len(train_loader) * epochs
        scheduler_3 = get_linear_schedule_with_warmup(opt3, total_steps // 10, max(total_steps, 1))

        best_map3 = 0.0
        best_state3 = None

        for epoch in range(epochs):
            model_3.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                opt3.zero_grad()
                outputs = model_3(input_ids, attention_mask)
                loss = criterion_3(outputs, labels)
                loss.backward()
                opt3.step()
                scheduler_3.step()

                train_loss += loss.item()

            model_3.eval()
            predictions = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                    outputs = model_3(input_ids, attention_mask)
                    predictions.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)

            top3_indices = np.argsort(-predictions, axis=1)[:, :3]
            val_preds = [[options[idx] for idx in indices] for indices in top3_indices]

            actual = train.iloc[val_idx]['answer'].tolist()
            val_map3 = mapk(actual, val_preds)

            if val_map3 >= best_map3:
                best_map3 = val_map3
                best_state3 = {k: v.cpu().clone() for k, v in model_3.state_dict().items()}

        if best_state3 is not None:
            model_3.load_state_dict(best_state3)

        model_3.eval()
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, _ = batch
                val_preds.append(model_3(input_ids.to(device), attention_mask.to(device)).cpu().numpy())

        train_minilm_oof[val_idx] = np.concatenate(val_preds, axis=0)

        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = batch
                test_preds.append(model_3(input_ids.to(device), attention_mask.to(device)).cpu().numpy())

        test_minilm_matrix += np.concatenate(test_preds, axis=0) / 3.0

        print(f"Model 3 (MiniLM) Fold {fold+1}/3 Best MAP@3: {best_map3:.4f}")

        clear_vram(['model_3', 'opt3', 'scheduler_3', 'train_dataset', 'val_dataset', 'test_dataset'])

    return train_minilm_oof, test_minilm_matrix
