"""
Model 3: MiniLM-based MCQ Classifier.
Uses MiniLM-L12 as a bi-encoder.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from src.utils import set_seed, mapk, clear_vram

logger = logging.getLogger(__name__)


class MiniLMDataset(Dataset):
    """Dataset for MiniLM model."""
    
    def __init__(self, df, tokenizer, options, max_length=128, is_test=False):
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
            text = f"{row['prompt']} {row[option]}"
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encoding['input_ids'].squeeze(0))
            attention_masks.append(encoding['attention_mask'].squeeze(0))
        
        if self.is_test:
            return torch.stack(input_ids), torch.stack(attention_masks)
        
        return (
            torch.stack(input_ids),
            torch.stack(attention_masks),
            torch.tensor(self.options.index(row['answer']), dtype=torch.long)
        )


class MiniLMModel(nn.Module):
    """MiniLM-based MCQ classifier using [CLS] token."""
    
    def __init__(self, model_name: str):
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


def train_minilm(
    train_df,
    test_df,
    options: List[str],
    config: Dict[str, Any],
    device: torch.device,
    n_folds: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Train MiniLM model with cross-validation.
    
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
    seed = config.get('seed', 1)
    model_name = config.get('name', 'microsoft/MiniLM-L12-H384-uncased')
    max_length = config.get('max_length', 128)
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 3)
    lr = config.get('learning_rate', 2e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
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
        train_dataset = MiniLMDataset(train_df.iloc[train_idx], tokenizer, options, max_length)
        val_dataset = MiniLMDataset(train_df.iloc[val_idx], tokenizer, options, max_length)
        test_dataset = MiniLMDataset(test_df, tokenizer, options, max_length, is_test=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MiniLMModel(model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        total_steps = len(train_loader) * epochs
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
            
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
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
        
        logger.info(f"MiniLM Fold {fold+1}/{n_folds} Best MAP@3: {best_map3:.4f}")
        
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
