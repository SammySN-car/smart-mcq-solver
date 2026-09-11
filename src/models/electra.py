import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from src.utils import set_seed, mapk, clear_vram


class CustomDataset2(Dataset):
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
            enc = self.tokenizer(
                str(row['prompt']), str(row[option]),
                max_length=256, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids.append(enc['input_ids'].squeeze(0))
            attention_masks.append(enc['attention_mask'].squeeze(0))

        if self.is_test:
            return torch.stack(input_ids), torch.stack(attention_masks)

        return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(self.options.index(row['answer']), dtype=torch.long)


class ELECTRAMCQModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.electra = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        b, n, l = input_ids.size()
        flat_ids = input_ids.view(b * n, l)
        flat_mask = attention_mask.view(b * n, l)

        outputs = self.electra(input_ids=flat_ids, attention_mask=flat_mask)

        mask_expanded = flat_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        logits = torch.mean(torch.stack([self.classifier(drop(pooled)) for drop in self.dropouts], dim=0), dim=0)

        return logits.view(b, n)


def get_llrd_optimizer_electra(model, base_lr=1e-5, weight_decay=0.01, decay_rate=0.9):
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    parameters = []

    if hasattr(model.electra, 'encoder') and hasattr(model.electra.encoder, 'layer'):
        layers = model.electra.encoder.layer
    else:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    n_layers = len(layers)

    embed_lr = base_lr * (decay_rate ** n_layers)
    parameters += [
        {'params': [p for n, p in model.electra.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': embed_lr},
        {'params': [p for n, p in model.electra.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': embed_lr}
    ]

    for i, layer in enumerate(layers):
        layer_lr = base_lr * (decay_rate ** (n_layers - i - 1))
        parameters += [
            {'params': [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay, 'lr': layer_lr},
            {'params': [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': layer_lr}
        ]

    parameters += [{'params': model.classifier.parameters(), 'weight_decay': weight_decay, 'lr': base_lr}]

    return torch.optim.AdamW(parameters, eps=1e-6)


def train_electra(train, test, options, model_name, config, device, n_folds=5):
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 6)
    accumulation_steps = config.get('accumulation_steps', 4)

    n_train = len(train)
    n_test = len(test)

    test_electra_matrix = np.zeros((n_test, 5))
    train_electra_oof = np.zeros((n_train, 5))

    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf2.split(np.arange(n_train), train['answer'].values)):
        set_seed(seed + fold)

        train_dataset = CustomDataset2(train.iloc[train_idx], model_name, options)
        val_dataset = CustomDataset2(train.iloc[val_idx], model_name, options)
        test_dataset = CustomDataset2(test, model_name, options, is_test=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model_2 = ELECTRAMCQModel(model_name).to(device)
        opt2 = get_llrd_optimizer_electra(model_2, base_lr=1e-5, weight_decay=0.01, decay_rate=0.9)
        criterion_2 = nn.CrossEntropyLoss()

        total_steps = (len(train_loader) // accumulation_steps) * epochs
        scheduler_2 = get_linear_schedule_with_warmup(opt2, total_steps // 10, max(total_steps, 1))

        best_map2 = 0.0
        best_state2 = None

        for epoch in range(epochs):
            model_2.train()
            train_loss = 0.0
            opt2.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model_2(input_ids, attention_mask)
                loss = criterion_2(outputs, labels) / accumulation_steps
                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model_2.parameters(), 1.0)
                    opt2.step()
                    scheduler_2.step()
                    opt2.zero_grad()

                train_loss += loss.item() * accumulation_steps

            model_2.eval()
            predictions = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                    outputs = model_2(input_ids, attention_mask)
                    predictions.append(outputs.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)

            top3_indices = np.argsort(-predictions, axis=1)[:, :3]
            val_preds = [[options[idx] for idx in indices] for indices in top3_indices]

            actual = train.iloc[val_idx]["answer"].tolist()
            val_map3 = mapk(actual, val_preds)

            if val_map3 >= best_map2:
                best_map2 = val_map3
                best_state2 = {k: v.cpu().clone() for k, v in model_2.state_dict().items()}

        if best_state2 is not None:
            model_2.load_state_dict(best_state2)

        model_2.eval()
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                val_preds.append(model_2(input_ids, attention_mask).cpu().numpy())

        train_electra_oof[val_idx] = np.concatenate(val_preds, axis=0)

        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = batch
                test_preds.append(model_2(input_ids.to(device), attention_mask.to(device)).cpu().numpy())

        test_electra_matrix += np.concatenate(test_preds, axis=0) / float(n_folds)

        print(f"Model 2 (ELECTRA) Fold {fold+1}/{n_folds} Best MAP@3: {best_map2:.4f}")

        clear_vram(['model_2', 'opt2', 'scheduler_2', 'train_dataset', 'val_dataset', 'test_dataset'])

    return train_electra_oof, test_electra_matrix
