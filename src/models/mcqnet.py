import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import get_linear_schedule_with_warmup
from src.utils import set_seed, mapk, clear_vram


class CustomDataset1(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class MCQNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_mcqnet(train, test, options, config, device, data_dir, n_folds=5):
    from src.data import preprocess_text, create_tfidf_features

    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 40)
    lr = config.get('learning_rate', 1e-3)

    n_train = len(train)
    n_test = len(test)

    test_tfidf_matrix = np.zeros((n_test, 5))
    train_tfidf_oof = np.zeros((n_train, 5))

    skf1 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf1.split(np.arange(n_train), train['answer'].values)):
        set_seed(seed + fold)

        preprocessed_train = preprocess_text(train.iloc[train_idx], options)
        preprocessed_val = preprocess_text(train.iloc[val_idx], options)
        test_preprocessed = preprocess_text(test, options)

        x_train, x_val, y_train, x_test = create_tfidf_features(
            preprocessed_train, preprocessed_val, train.iloc[train_idx], test_preprocessed, options
        )

        y_val = np.array([1.0 if ans == opt else 0.0
                          for ans in train.iloc[val_idx]['answer']
                          for opt in options], dtype=np.float32)

        train_dataset = CustomDataset1(x_train, y_train)
        val_dataset = CustomDataset1(x_val, y_val)
        test_dataset = CustomDataset1(x_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model_1 = MCQNet(input_dim=x_train.shape[1]).to(device)
        opt1 = torch.optim.Adam(model_1.parameters(), lr=lr)
        criterion_1 = nn.BCELoss()

        total_steps = len(train_loader) * epochs
        scheduler_1 = get_linear_schedule_with_warmup(opt1, total_steps // 10, total_steps)

        best_map3 = 0.0
        best_state = None

        for epoch in range(epochs):
            model_1.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                opt1.zero_grad()
                outputs = model_1(batch_X)
                loss = criterion_1(outputs, batch_y)
                loss.backward()
                opt1.step()
                scheduler_1.step()

                train_loss += loss.item()

            model_1.eval()
            val_preds_flat = []

            with torch.no_grad():
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(device)
                    val_preds_flat.append(model_1(batch_X).cpu().numpy())

            val_preds_flat = np.concatenate(val_preds_flat, axis=0)
            val_preds_matrix = val_preds_flat.reshape(-1, 5)

            top3_indices = np.argsort(-val_preds_matrix, axis=1)[:, :3]
            val_preds_letters = [[options[idx] for idx in indices] for indices in top3_indices]

            actual = train.iloc[val_idx]['answer'].tolist()
            val_map3 = mapk(actual, val_preds_letters)

            if val_map3 >= best_map3:
                best_map3 = val_map3
                best_state = {k: v.cpu().clone() for k, v in model_1.state_dict().items()}

        if best_state is not None:
            model_1.load_state_dict(best_state)

        model_1.eval()
        val_preds_flat = []

        with torch.no_grad():
            for batch_X, _ in val_loader:
                val_preds_flat.append(model_1(batch_X.to(device)).cpu().numpy())

        train_tfidf_oof[val_idx] = np.concatenate(val_preds_flat, axis=0).reshape(-1, 5)

        test_preds_flat = []

        with torch.no_grad():
            for batch_X in test_loader:
                test_preds_flat.append(model_1(batch_X.to(device)).cpu().numpy())

        test_tfidf_matrix += np.concatenate(test_preds_flat, axis=0).reshape(-1, 5) / float(n_folds)

        print(f"Model 1 (TF-IDF MCQNet) Fold {fold+1}/{n_folds} Best MAP@3: {best_map3:.4f}")

        clear_vram(['model_1', 'opt1', 'scheduler_1', 'train_dataset', 'val_dataset', 'test_dataset'])

    return train_tfidf_oof, test_tfidf_matrix
