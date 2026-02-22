import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy

def loss_fn(y_true, y_pred):
    """
    Negative log-likelihood loss for log-normal distribution.
    y_true: tensor of shape [batch, 1]
    y_pred: tensor of shape [batch, 2], first column σ, second column s -mu in log-space- (shape, scale)
    """
    assert y_true.size(1) == 1, "y_true must be of shape [batch, 1]"
    assert y_pred.size(1) == 2, "y_pred must be of shape [batch, 2]"

    y_true = y_true.to(y_pred.device)
    s = y_pred[:, 0:1]
    scale = y_pred[:, 1:2]
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)
    help1 = 0.5 * ((log_true - log_scale) / s) ** 2
    lh = -torch.log(s) - log_true - help1
    return -lh.mean()

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
        x = self.tanh(self.bn1(self.fc1(x)))
        x = self.tanh(self.bn2(self.fc2(x)))
        return torch.exp(self.out(x))

class DistNetModel:
    def __init__(
        self,
        n_input_features,
        n_epohcs,
        batch_size,
        wc_time_limit,
        save_path=None,
        X_valid=None,
        y_valid=None,
        early_stopping=False,
        early_stopping_patience=50,
    ):
        # as in the original DistNet paper
        self.n_epochs = n_epohcs
        self.wc_time_limit = wc_time_limit  # seconds
        self.batch_size = batch_size
        self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        if save_path is None:
            self.save_flag = False
        else:
            self.save_path = save_path
            self.save_flag = True

        # validation data - direct device placement
        if X_valid is not None and y_valid is not None:
            self.X_valid = torch.as_tensor(X_valid, dtype=torch.float32, device=self.device)
            self.y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=self.device)
            self.validation_available = True
        else:
            self.validation_available = False

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        if self.early_stopping:
            assert self.validation_available, "Early stopping requires validation data"
            self.best_model_checkpoint = None
            self.best_val_loss = float('inf')
            self.epochs_no_improve = 0

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
        gamma = (final_lr / initial_lr) ** (1.0 / float(self.n_epochs))
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
        instance = cls.__new__(cls)
        instance.device = torch.device('cpu')
        instance.save_path = load_path
        instance.model = DistNet(n_input_features).to(instance.device)
        instance.model.load_state_dict(torch.load(load_path, map_location=instance.device))
        instance.model.eval()
        return instance
    
    def train(self, X_train, y_train):
        assert X_train.ndim == 2, "X_train must have batch dimension"
        assert y_train.ndim == 2, "y_train must have batch dimension"
        self.model.train()
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        n_samples = X.size(0)

        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            epoch_loss = 0.0
            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start : (start + self.batch_size)]
                bx, by = X[idx], y[idx]
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = loss_fn(by, preds)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * bx.size(0)

            self.scheduler.step()
            avg_train_loss = epoch_loss / n_samples
            elapsed = time.time() - start_time

            if self.validation_available:
                assert self.X_valid is not None and self.y_valid is not None, "Validation data not provided"
                self.model.eval()
                with torch.no_grad():
                    vpred = self.model(self.X_valid)
                    val_loss = loss_fn(self.y_valid, vpred)
                # Early stopping check
                if self.early_stopping:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_checkpoint = copy.deepcopy(self.model.state_dict())
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1

                    if self.epochs_no_improve >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss:.4f}")
                        break

                print(f"Epoch {epoch}/{self.n_epochs} | Train {avg_train_loss:.4f} | Val {val_loss:.4f} | LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")
                self.model.train()

            else:
                if epoch == 1 or epoch % 100 == 0:
                    print(f"Epoch {epoch}/{self.n_epochs} | Loss {avg_train_loss:.4f} | LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

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

    def predict(self, X):
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()
