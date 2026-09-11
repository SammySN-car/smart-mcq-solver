"""
Model 1: TF-IDF based MCQNet.
Custom neural network for MCQ classification using TF-IDF features.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup

from src.utils import set_seed, mapk, clear_vram

logger = logging.getLogger(__name__)


class MCQDataset(Dataset):
    """Dataset for TF-IDF features."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class MCQNet(nn.Module):
    """Custom neural network for MCQ classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 64, 1]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer with sigmoid
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(1)


def train_mcqnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    train_answers: List[str],
    val_answers: List[str],
    options: List[str],
    config: Dict[str, Any],
    device: torch.device,
    n_folds: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Train MCQNet with cross-validation.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        train_answers: Ground truth answers for training set.
        val_answers: Ground truth answers for validation set.
        options: List of option letters.
        config: Model configuration.
        device: Torch device.
        n_folds: Number of CV folds.
    
    Returns:
        Tuple of (train_oof, test_predictions).
    """
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 40)
    lr = config.get('learning_rate', 0.001)
    hidden_dims = config.get('hidden_dims', [256, 64, 1])
    dropout = config.get('dropout', 0.3)
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    test_matrix = np.zeros((n_test, 5))
    train_oof = np.zeros((n_train, 5))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(n_train), train_answers)):
        set_seed(seed + fold)
        
        # Create datasets
        train_dataset = MCQDataset(X_train[train_idx], y_train[train_idx])
        val_dataset = MCQDataset(X_train[val_idx], y_train[val_idx])
        test_dataset = MCQDataset(X_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MCQNet(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        best_map3 = 0.0
        best_state = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(device)
                    val_preds.append(model(batch_X).cpu().numpy())
            
            val_preds = np.concatenate(val_preds, axis=0).reshape(-1, 5)
            top3_indices = np.argsort(-val_preds, axis=1)[:, :3]
            val_preds_letters = [[options[idx] for idx in indices] for indices in top3_indices]
            
            val_map3 = mapk(val_answers, val_preds_letters)
            
            if val_map3 >= best_map3:
                best_map3 = val_map3
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best model and generate OOF predictions
        if best_state is not None:
            model.load_state_dict(best_state)
            model.eval()
            
            val_preds = []
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    val_preds.append(model(batch_X.to(device)).cpu().numpy())
            
            train_oof[val_idx] = np.concatenate(val_preds, axis=0).reshape(-1, 5)
            
            # Test predictions
            test_preds = []
            with torch.no_grad():
                for batch_X in test_loader:
                    test_preds.append(model(batch_X.to(device)).cpu().numpy())
            
            test_matrix += np.concatenate(test_preds, axis=0).reshape(-1, 5) / float(n_folds)
        
        logger.info(f"MCQNet Fold {fold+1}/{n_folds} Best MAP@3: {best_map3:.4f}")
        
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
