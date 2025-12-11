"""
Prediction Models Module
Implements Random Forest, LSTM, and Transformer models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestPipeline:
    """Random Forest for classification and regression tasks"""

    def __init__(self, config: Dict, task: str = 'classification'):
        self.config = config
        self.task = task  # 'classification' or 'regression'
        self.model = None
        self.label_encoder = LabelEncoder() if task == 'classification' else None

    def prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str] = None) -> Tuple:
        """Prepare data for training"""
        if feature_cols is None:
            # Auto-select numeric columns
            exclude_cols = ['function_id', 'function_name', 'file_path', 'repo_name',
                            'start_line', 'end_line', 'code', 'operators', 'operands', target_col]
            feature_cols = [col for col in df.columns
                            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        X = df[feature_cols].fillna(0).values
        y = df[target_col].values

        if self.task == 'classification' and self.label_encoder:
            y = self.label_encoder.fit_transform(y)

        return train_test_split(X, y, test_size=self.config['evaluation']['test_size'],
                                random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest with hyperparameter tuning"""
        logger.info("Training Random Forest")

        param_grid = self.config['models']['prediction']['random_forest']

        if self.task == 'classification':
            base_model = RandomForestClassifier(random_state=param_grid['random_state'])
        else:
            base_model = RandomForestRegressor(random_state=param_grid['random_state'])

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid={
                'n_estimators': param_grid['n_estimators'],
                'max_depth': param_grid['max_depth'],
                'min_samples_split': param_grid['min_samples_split']
            },
            cv=self.config['evaluation']['cv_folds'],
            scoring='accuracy' if self.task == 'classification' else 'r2',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test set"""
        logger.info("Evaluating Random Forest")

        y_pred = self.model.predict(X_test)

        if self.task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            # Class-wise metrics
            if self.label_encoder:
                class_names = self.label_encoder.classes_
                report = classification_report(y_test, y_pred, target_names=class_names,
                                                output_dict=True, zero_division=0)
                metrics['classification_report'] = report
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        # Feature importance
        feature_importance = self.model.feature_importances_
        metrics['feature_importance'] = feature_importance.tolist()

        return metrics

    def get_feature_importance(self, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
        """Get top k important features"""
        if self.model is None:
            return None

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_k)

        return importance_df


class CodeDataset(Dataset):
    """PyTorch Dataset for code features"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    """LSTM model for sequence classification"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Reshape: (batch, features) -> (batch, seq_len=1, features)
        x = x.unsqueeze(1)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        out = self.dropout(h_n[-1])
        out = self.fc(out)

        return out


class LSTMPipeline:
    """LSTM training and evaluation pipeline"""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str] = None) -> Tuple:
        """Prepare data for LSTM"""
        if feature_cols is None:
            exclude_cols = ['function_id', 'function_name', 'file_path', 'repo_name',
                            'start_line', 'end_line', 'code', 'operators', 'operands', target_col]
            feature_cols = [col for col in df.columns
                            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        X = df[feature_cols].fillna(0).values
        y = df[target_col].values
        y = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['evaluation']['test_size'], random_state=42
        )

        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, len(feature_cols), len(self.label_encoder.classes_)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, input_size: int, num_classes: int) -> Dict[str, Any]:
        """Train LSTM model"""
        logger.info("Training LSTM")

        lstm_config = self.config['models']['prediction']['lstm']

        # Create model
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            num_classes=num_classes,
            dropout=lstm_config['dropout']
        ).to(self.device)

        # Create data loader
        train_dataset = CodeDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=lstm_config['batch_size'],
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lstm_config['learning_rate'])

        # Training loop
        history = {'train_loss': [], 'train_acc': []}

        for epoch in range(lstm_config['epochs']):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = correct / total

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{lstm_config['epochs']}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate LSTM model"""
        logger.info("Evaluating LSTM")

        self.model.eval()

        test_dataset = CodeDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)

                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }

        return metrics


class TransformerPipeline:
    """Transformer-based code classification using CodeBERT"""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()

    def prepare_model(self, num_classes: int):
        """Initialize CodeBERT model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_name = self.config['models']['prediction']['transformer']['model_name']

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes
            )

            logger.info(f"Loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise

    def train(self, df: pd.DataFrame, target_col: str, code_col: str = 'code') -> Dict[str, Any]:
        """Train transformer model"""
        logger.info("Training Transformer (CodeBERT)")

        # Prepare labels
        y = df[target_col].values
        y = self.label_encoder.fit_transform(y)

        # Initialize model
        num_classes = len(self.label_encoder.classes_)
        self.prepare_model(num_classes)

        # Note: Full transformer training requires more setup
        # This is a simplified placeholder
        logger.info("Transformer training would require more resources and setup")
        logger.info("Consider using pre-trained embeddings + classifier instead")

        return {'status': 'placeholder'}

    def evaluate(self, df: pd.DataFrame, target_col: str, code_col: str = 'code') -> Dict[str, Any]:
        """Evaluate transformer model"""
        logger.info("Transformer evaluation placeholder")
        return {'status': 'placeholder'}


if __name__ == "__main__":
    import yaml

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Example usage
    logger.info("Prediction models module loaded")
