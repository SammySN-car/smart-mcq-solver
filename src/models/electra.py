"""
Model 2: ELECTRA-based MCQ Classifier.
Uses ELECTRA-base with LLRD and multi-sample dropout.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup

from src.utils import set_seed, mapk, clear_vram

logger = logging.getLogger(__name__)


class ElectraDataset(Dataset):
    """Dataset for ELECTRA model."""
    
    def __init__(self, df, tokenizer, options, max_length=256, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.options = options
        self.max_length = max_length
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, attention_masks = [], []
        
        for option in self.options:
            enc = self.tokenizer(
                str(row['prompt']),
                str(row[option]),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(enc['input_ids'].squeeze(0))
            attention_masks.append(enc['attention_mask'].squeeze(0))
        
        if self.is_test:
            return torch.stack(input_ids), torch.stack(attention_masks)
        
        return (
            torch.stack(input_ids),
            torch.stack(attention_masks),
            torch.tensor(self.options.index(row['answer']), dtype=torch.long)
        )


class ElectraMCQModel(nn.Module):
    """ELECTRA-based MCQ classifier with multi-sample dropout."""
    
    def __init__(self, model_name: str, n_dropout_heads: int = 5, dropout_rate: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.electra = AutoModel.from_pretrained(model_name, config=self.config, trust_remote_code=True)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_dropout_heads)])
        self.classifier = nn.Linear(self.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        b, n, l = input_ids.size()
        flat_ids = input_ids.view(b * n, l)
        flat_mask = attention_mask.view(b * n, l)
        
        outputs = self.electra(input_ids=flat_ids, attention_mask=flat_mask)
        
        # Mean pooling
        mask_expanded = flat_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        # Multi-sample dropout
        logits = torch.mean(
            torch.stack([drop(pooled) for drop in self.dropouts], dim=0),
            dim=0
        )
        
        return logits.view(b, n)


def get_llrd_optimizer(model, base_lr=1e-5, weight_decay=0.01, decay_rate=0.9):
    """Get optimizer with Layerwise Learning Rate Decay."""
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    parameters = []
    
    if hasattr(model.electra, 'encoder') and hasattr(model.electra.encoder, 'layer'):
        layers = model.electra.encoder.layer
    else:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    n_layers = len(layers)
    
    # Embedding layer
    embed_lr = base_lr * (decay_rate ** n_layers)
    parameters.extend([
        {
            'params': [p for n, p in model.electra.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
            'lr': embed_lr
        },
        {
            'params': [p for n, p in model.electra.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': embed_lr
        }
    ])
    
    # Encoder layers
    for i, layer in enumerate(layers):
        layer_lr = base_lr * (decay_rate ** (n_layers - i - 1))
        parameters.extend([
            {
                'params': [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': layer_lr
            },
            {
                'params': [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': layer_lr
            }
        ])
    
    # Classifier
    parameters.append({
        'params': model.classifier.parameters(),
        'weight_decay': weight_decay,
        'lr': base_lr
    })
    
    return torch.optim.AdamW(parameters, eps=1e-6)


def train_electra(
    train_df,
    test_df,
    options: List[str],
    config: Dict[str, Any],
    device: torch.device,
    n_folds: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Train ELECTRA model with cross-validation.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        options: List of option letters.
        config: Model configuration.
        device: Torch device.
        n_folds: Number of CV folds.
    
    Returns:
        Tuple of (train_oof, test_predictions).
    """
    seed = config.get('seed', 42)
    model_name = config.get('name', 'google/electra-base-discriminator')
    max_length = config.get('max_length', 256)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 6)
    lr = config.get('learning_rate', 1e-5)
    weight_decay = config.get('weight_decay', 0.01)
    decay_rate = config.get('decay_rate', 0.9)
    n_dropout_heads = config.get('n_dropout_heads', 5)
    dropout_rate = config.get('dropout_rate', 0.2)
    accumulation_steps = config.get('accumulation_steps', 4)
    
    n_train = len(train_df)
    n_test = len(test_df)
    
    test_matrix = np.zeros((n_test, 5))
    train_oof = np.zeros((n_train, 5))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(n_train), train_df['answer'].values)):
        set_seed(seed + fold)
        
        # Create datasets
        train_dataset = ElectraDataset(train_df.iloc[train_idx], tokenizer, options, max_length)
        val_dataset = ElectraDataset(train_df.iloc[val_idx], tokenizer, options, max_length)
        test_dataset = ElectraDataset(test_df, tokenizer, options, max_length, is_test=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = ElectraMCQModel(model_name, n_dropout_heads, dropout_rate).to(device)
        optimizer = get_llrd_optimizer(model, lr, weight_decay, decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_steps = (len(train_loader) // accumulation_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=max(total_steps, 1)
        )
        
        best_map3 = 0.0
        best_state = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for step, batch in enumerate(train_loader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels) / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
            
            # Validation
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    predictions.append(outputs.cpu().numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            top3_indices = np.argsort(-predictions, axis=1)[:, :3]
            val_preds = [[options[idx] for idx in indices] for indices in top3_indices]
            
            val_answers = train_df.iloc[val_idx]['answer'].tolist()
            val_map3 = mapk(val_answers, val_preds)
            
            if val_map3 >= best_map3:
                best_map3 = val_map3
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best model and generate OOF predictions
        if best_state is not None:
            model.load_state_dict(best_state)
            model.eval()
            
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, _ = batch
                    val_preds.append(model(input_ids.to(device), attention_mask.to(device)).cpu().numpy())
            
            train_oof[val_idx] = np.concatenate(val_preds, axis=0)
            
            # Test predictions
            test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask = batch
                    test_preds.append(model(input_ids.to(device), attention_mask.to(device)).cpu().numpy())
            
            test_matrix += np.concatenate(test_preds, axis=0) / float(n_folds)
        
        logger.info(f"ELECTRA Fold {fold+1}/{n_folds} Best MAP@3: {best_map3:.4f}")
        
        # Clear memory
        clear_vram({
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        })
    
    return train_oof, test_matrix
