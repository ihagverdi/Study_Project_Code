import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

from tabpfn_project.helper.utils import TargetScale

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def loss_fn_max(y_true, y_pred):
    """
    Negative log-likelihood loss for log-normal distribution.
    y_true: tensor of shape [batch, 1]
    y_pred: tensor of shape [batch, 2]
    """
    assert y_true.size(1) == 1, "y_true must be of shape [batch, 1]"
    assert y_pred.size(1) == 2, "y_pred must be of shape [batch, 2]"
    assert len(y_true) == len(y_pred), "loss_fn: y_true and y_pred must have the same batch size"

    y_true = y_true.to(y_pred.device)
    shape = y_pred[:, 0:1]
    scale = y_pred[:, 1:2]  # e^mu
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)
    help1 = 0.5 * (((log_true - log_scale) / shape) ** 2)
    lh = -torch.log(shape) - log_true - help1
    return -lh.mean()

def loss_fn_log(y_true, y_pred):
    """
    Negative log-likelihood loss for Normal distribution.
    Used when targets are log-scaled.
    y_true: tensor of shape [batch, 1]
    y_pred: tensor of shape [batch, 2] -> [mu, sigma]
    """
    assert y_true.size(1) == 1, "y_true must be of shape [batch, 1]"
    assert y_pred.size(1) == 2, "y_pred must be of shape [batch, 2]"
    assert len(y_true) == len(y_pred), "loss_fn_log: y_true and y_pred must have the same batch size"

    y_true = y_true.to(y_pred.device)
    mu = y_pred[:, 0:1]    
    sigma = y_pred[:, 1:2] 
    
    help1 = 0.5 * (((y_true - mu) / sigma) ** 2)
    nll = torch.log(sigma) + help1
    return nll.mean()

class DistNet(nn.Module):
    def __init__(self, n_input_features):
        super(DistNet, self).__init__()
        self.fc1 = nn.Linear(n_input_features, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        For max-scale: outputs [shape, scale]
        For log-scale: outputs [mu, sigma]
        '''
        x = self.tanh(self.bn1(self.fc1(x)))
        x = self.tanh(self.bn2(self.fc2(x)))
        return torch.exp(self.out(x))

class DistNetModel:
    def __init__(
        self,
        model_target_scale,
        n_input_features,
        n_epochs,
        batch_size,
        wc_time_limit,
        random_state,
        save_path=None,
        X_valid=None,
        y_valid=None,
        early_stopping=False,
        early_stopping_patience=20,
    ):
        set_seed(random_state)
        self.model_target_scale = model_target_scale
        self.n_epochs = n_epochs
        self.expected_n_epochs = n_epochs
        self.wc_time_limit = wc_time_limit  # seconds
        self.batch_size = batch_size
        self.device = torch.device('cpu')

        self.save_flag = save_path is not None
        if self.save_flag:
            self.save_path = save_path

        # validation data - direct device placement
        self.validation_available = (X_valid is not None and y_valid is not None)
        if self.validation_available:
            assert X_valid.ndim == 2 and y_valid.ndim == 2 and y_valid.shape[-1] == 1
            self.X_valid = torch.as_tensor(X_valid, dtype=torch.float32, device=self.device)
            self.y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=self.device)

        self.early_stopping = early_stopping
        if self.early_stopping:
            self.early_stopping_patience = early_stopping_patience
            assert self.validation_available, "Early stopping requires validation data"
            self.best_model_checkpoint = None
            self.best_val_loss = float('inf')
            self.epochs_no_improve = 0
            self.best_epoch = 0  # epoch index where validation loss was minimised

        # model, optimizer, scheduler
        self.model = DistNet(n_input_features).to(self.device)
        
        initial_lr = 1e-3
        final_lr = 1e-5
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=initial_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        gamma = (final_lr / initial_lr) ** (1.0 / float(self.expected_n_epochs))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    
    @classmethod
    def load_model(cls, load_path, n_input_features):
        """
        Load a pre-trained model for inference.
        
        Args:
            load_path: Path to the saved .pt file
            n_input_features: Number of input features (must match training)
        
        Returns:
            DistNetModel instance with loaded weights
        """
        model = DistNet(n_input_features)
        model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
        model.eval()
        
        # Create a minimal instance for inference
        instance = cls.__new__(cls)
        instance.model = model
        instance.device = torch.device('cpu')
        
        return instance
    
    def train(self, X_train, y_train):
        assert X_train.ndim == 2 and y_train.ndim == 2, "X_train and y_train must have batch dimension"
        assert len(X_train) == len(y_train), "X_train and y_train must have same length"
        assert y_train.shape[1] == 1, "y_train must have shape [batch, 1]"

        self.model.train()
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

        n_samples = X.size(0)
        start_time = time.time()

        criterion = loss_fn_log if self.model_target_scale == TargetScale.LOG else loss_fn_max
        
        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start : (start + self.batch_size)]

                if len(idx) == 1:
                    continue  # Skip batches of size 1
                
                bx, by = X[idx], y[idx]
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(by, preds)
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1e-2)
                self.optimizer.step()
                epoch_loss += loss.item() * bx.size(0)

            self.scheduler.step()
            avg_train_loss = epoch_loss / n_samples
            elapsed = time.time() - start_time

            val_loss = None
            if self.validation_available:
                self.model.eval()
                with torch.inference_mode():
                    vpred = self.model(self.X_valid)
                    val_loss = criterion(self.y_valid, vpred)
                # Early stopping check
                if self.early_stopping:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_checkpoint = deepcopy(self.model.state_dict())
                        self.best_epoch = epoch
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1

                    if self.epochs_no_improve >= self.early_stopping_patience:
                        print(f"Early stopping | Best epoch: {epoch-self.early_stopping_patience}. Best validation loss: {self.best_val_loss:.4f}")
                        break
                self.model.train()

            if epoch == 1 or epoch % 100 == 0:
                val_str = f"{val_loss:.4f}" if val_loss is not None else "Unavailable"
                print(f"Epoch {epoch}/{self.n_epochs} | Train Loss {avg_train_loss:.4f} | Validation Loss {val_str} LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

            if elapsed > self.wc_time_limit:
                print(f"Time limit reached ({elapsed:.1f}s), stopping training.")
                break
        
        if self.early_stopping:
            assert self.best_model_checkpoint is not None, "No best model checkpoint has been found"
            # Restore best model
            print(f"Restoring best model with val_loss: {self.best_val_loss:.4f}")
            self.model.load_state_dict(self.best_model_checkpoint)

        if self.save_flag:
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")

    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            return self.model(X_test_t).cpu().numpy()
