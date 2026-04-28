from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F

# Constants
EPSILON = 1e-10
TWO_PI = 2 * np.pi
LOG_2PI = 1.837877066
LOG_2PI = np.log(TWO_PI)
SQRT_TWO = np.sqrt(2.0)
HALF = 0.5
E = 2.71828182845904523536028

training_hparams = {
    "n_epochs": 250,
    "n_expected_epochs": 500,
    "n_optim": "SGD",
    "batch_size": 32,
    "start_rate": 1e-3,
    "end_rate": 1e-5,
    "clip_gradient_norm": 1e-2,
    "split_ration": 0.8,
    "seed": 0,
    "n_ens": 16,
    "beta_type": 0.1,
    "split_ratio": 0.2,
    "early_stop": 20
}

model_hparams = {
    "n_fcdepth": 16,
    "output_size": 1,
    "drop_value": 0.0,
    "posterior_mu1": 0.0,
    "posterior_mu2": 0.1,
    "posterior_sigma1": -3.0,
    "posterior_sigma2": 0.1,
    "mixture_pi": 0.5,
    "prior_sigma1": 0.3,
    "prior_sigma2": 0.01
}

def lognorm_bayesian(outputs, observation, reduce=True):
    """
    Bayesian Lognormal loss for UNCENSORED data only.
    
    Args:
        outputs (torch.Tensor): shape (batch_size, num_samples)
        observation (torch.Tensor): shape (batch_size, 1) - observed values only
        reduce (bool): Whether to reduce to mean or not
    """
    # MLE of sample for each instance
    flag = outputs.sum(dim=1) > 1e-8
    outputs = outputs[flag]
    observation = observation[flag]
    outputs += 1e-10
    
    mu = torch.exp(torch.log(outputs).mean(dim=1, keepdim=True))
    sigma = torch.log(outputs).std(dim=1, keepdim=True)
    target = observation[:, 0:1] + EPSILON
    
    # Compute PDF for all samples (no censoring split)
    pdf_help1 = HALF * ((torch.log(target) - torch.log(mu)) / sigma)**2
    llh = torch.flatten(-torch.log(target) - torch.log(sigma) - pdf_help1)
    
    return torch.mean(-llh) if reduce else -llh

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - batch_idx) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

class BBBLinear2(ModuleWrapper):
    
    def __init__(self, in_features, out_features, model_config, device=None, bias=False, name='BBBLinear'):
        super(BBBLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name

        self.prior_mu = 0.0
        self.prior_sigma = 0.1
        self.prior_sigma1 = model_config['prior_sigma1']
        self.prior_sigma2 = model_config['prior_sigma2']
        self.pi = model_config['mixture_pi']

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        self.posterior_mu1 = model_config['posterior_mu1']
        self.posterior_mu2 = model_config['posterior_mu2']
        self.posterior_sigma1 = model_config['posterior_sigma1']
        self.posterior_sigma2 = model_config['posterior_sigma2']

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
        self.W_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

        if self.use_bias:
            self.bias_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
            self.bias_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

    def gaussian_prior(self, x, mu, sigma):
        PI = 3.1415926535897
        scaling = 1.0 / np.sqrt(2.0 * PI * (sigma ** 2))
        bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell

    def gaussian(self, x, mu, sigma):
        PI = 3.1415926535897
        scaling = 1.0 / torch.sqrt(2.0 * PI * (sigma ** 2))
        bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
        return scaling * bell

    def log_prior_sum(self, x):
        first_gaussian = self.pi * self.gaussian_prior(x, self.prior_mu, self.prior_sigma1)
        second_gaussian = (1 - self.pi) * self.gaussian_prior(x, self.prior_mu, self.prior_sigma2)
        return torch.log(first_gaussian + second_gaussian).sum()

    def log_variational_sum(self, x):
        return torch.log(self.gaussian(x, self.W_mu, self.W_sigma)).sum()

    def forward(self, x):
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        eps = torch.empty(act_mu.size(), device=act_mu.device).normal_(0, 1)
        return act_mu + act_std * eps

    def kl_loss(self):
        W_eps = torch.empty(self.W_mu.size(), device=self.W_mu.device).normal_(0, 1)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.weights = self.W_mu + W_eps * self.W_sigma
        kl = self.log_variational_sum(self.weights) - self.log_prior_sum(self.weights)
        return kl

class BayesDistNetFCN(ModuleWrapper):
    """
    Implementation of the modified DistNet model (Eggensperger et al, 2018)
    Bayesian layers are used in place of traditional dense layers.
    """

    def __init__(self, num_features, config: dict, name_suffix: str = ""):
        super().__init__()
        self.name_suffix = name_suffix

        # Model config
        self.n_fcdepth         = config['n_fcdepth']
        self.output_size       = config['output_size']
        self.drop_value        = config['drop_value']

        # Layer 1
        in_channel = num_features
        out_channel = self.n_fcdepth
        self.layer1 = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name='fc1'),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        # Layer 2
        in_channel = out_channel
        out_channel = self.n_fcdepth
        self.layer2 = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name='fc2'),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        # Layer 3: Final output from model
        in_channel = out_channel
        out_channel = self.output_size
        self.layer_end = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name='fc3'),
            # nn.BatchNorm1d(out_channel),
            nn.Softplus()
        )


    # Helper function for debugging
    def toStr(self) -> str:
        return "Bayes_Distnet" + "_" + self.name_suffix

def weights_init(m):
    """
    Initializes the weights for the given module using the Xavier Uniform method

    Args:
        m (torch.nn.Module): The module to initalize the weights
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()
        m.bias.data.fill_(0.01)

def count_parameters(model):
    """
    Counts the number of parameters for the given model

    Args:
        model (torch.nn.): The model to examine
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, optimizer, X_train, y_train, training_config, batch_size, device, epoch, num_epochs):
    training_loss = 0.0
    model.train()
    loss_fn = lognorm_bayesian
    n_ens = training_config['n_ens']
    clip_gradient_norm = training_config['clip_gradient_norm']
    beta_type = training_config['beta_type']
    total_kl = 0.0

    if not torch.is_tensor(X_train):
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    if not torch.is_tensor(y_train):
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    n_samples = X_train.size(0)
    indices = torch.randperm(n_samples, device=device)
    batches = []
    for start in range(0, n_samples, batch_size):
        idx = indices[start : (start + batch_size)]
        if idx.numel() <= 1:
            continue
        batches.append(idx)

    n_batches = len(batches)
    if n_batches == 0:
        raise ValueError("Batch size too large for training data.")

    for batch_idx, idx in enumerate(batches):
        inputs = X_train[idx]
        rts = y_train[idx]

        # zero the parameter gradients
        optimizer.zero_grad()

        # Propogate input to model and calculate loss
        if type(model) is BayesDistNetFCN:
            kl = 0.0
            batch_len = inputs.shape[0]
            outputs = torch.zeros(batch_len, n_ens, device=device)
            for j in range(n_ens):
                net_out, _kl = model(inputs)
                kl += _kl
                outputs[:,j] = net_out.flatten()
            kl /= n_ens

            beta = get_beta(batch_idx, n_batches, beta_type, epoch, num_epochs)
            loss = (kl * beta) / n_batches + loss_fn(outputs, rts, reduce=False).sum()
            total_kl += (kl * beta) / n_batches
        else:
            raise ValueError('Unknown net type.')

        # Propogate loss backwards and step optimizer
        loss.backward()

        # Clip the gradients
        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)
        # Step optimizer
        optimizer.step()

        training_loss += loss.item()

    return training_loss / n_batches, total_kl

def validate_model(model, X_valid, y_valid, training_config, batch_size, device, epoch=1, num_epochs=1, use_batching=False):
    loss_fn = lognorm_bayesian
    n_ens = training_config['n_ens']
    beta_type = training_config['beta_type']

    model.eval()

    validation_loss = 0.0
    log_loss = 0.0
    total_kl = 0.0
    if not torch.is_tensor(X_valid):
        X_valid = torch.as_tensor(X_valid, dtype=torch.float32, device=device)
    if not torch.is_tensor(y_valid):
        y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=device)

    n_samples = X_valid.size(0)

    if use_batching:
        indices = torch.arange(n_samples, device=device)
        batches = []
        for start in range(0, n_samples, batch_size):
            idx = indices[start : (start + batch_size)]
            if idx.numel() <= 1:
                continue
            batches.append(idx)

        n_batches = len(batches)
        if n_batches == 0:
            raise ValueError("Batch size too large for validation data.")
    else:
        if n_samples <= 1:
            raise ValueError("Batch size too large for validation data.")
        batches = [torch.arange(n_samples, device=device)]
        n_batches = 1

    with torch.no_grad():
        for batch_idx, idx in enumerate(batches):
            val_inputs = X_valid[idx]
            val_rts = y_valid[idx]

            if type(model) is BayesDistNetFCN:
                kl = 0.0
                batch_len = val_inputs.shape[0]
                outputs = torch.zeros(batch_len, n_ens, device=device)
                for j in range(n_ens):
                    net_out, _kl = model(val_inputs)
                    kl += _kl
                    outputs[:,j] = net_out.flatten()
                kl /= n_ens

                beta = get_beta(batch_idx, n_batches, beta_type, epoch, num_epochs)
                loss = loss_fn(outputs, val_rts, reduce=True).sum()
                val_loss = (kl * beta) / n_batches + loss
                total_kl += (kl * beta) / n_batches

            else:
                raise ValueError('Unknown net type.')

            validation_loss += val_loss.item()
            log_loss += loss.item()

            if loss != loss:
                torch.set_printoptions(edgeitems=10000)
                print(log_loss)
                print(outputs)
                print(kl)
                exit()

    return validation_loss / n_batches, log_loss / n_batches, total_kl

def train_bayesianDistNet(
    X_train_flat,
    y_train_flat,
    device,
    X_valid_flat=None,
    y_valid_flat=None,
    early_stopping=False,
    training_config=None,
    model_config=None,
):

    if training_config is None:
        training_config = training_hparams
    if model_config is None:
        model_config = model_hparams

    seed = training_config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    num_samples = X_train_flat.shape[0]
    num_features = X_train_flat.shape[1]
    model = BayesDistNetFCN(num_features, config=model_config)

    # Reset model and send model to device
    model.apply(weights_init)
    model.to(device)

    n_optim = training_config["n_optim"]
    start_rate = training_config["start_rate"]
    end_rate = training_config["end_rate"]
    n_expected_epochs = training_config["n_expected_epochs"]
    batch_size = training_config["batch_size"]
    n_epochs = training_config["n_epochs"]
    early_stopping_patience = training_config["early_stop"]

    # Optimizer and scheduler
    if n_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=start_rate, weight_decay=0.0001, momentum=0.95)
    elif n_optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=start_rate, weight_decay=0.001, amsgrad=False)
    else:
        raise ValueError('Unknown optimizer type.')
    
    decay_rate = np.exp(np.log(end_rate / start_rate) / n_expected_epochs)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    X_train_t = torch.as_tensor(X_train_flat, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train_flat, dtype=torch.float32, device=device)

    validation_available = (X_valid_flat is not None and y_valid_flat is not None)
    if validation_available:
        X_valid_t = torch.as_tensor(X_valid_flat, dtype=torch.float32, device=device)
        y_valid_t = torch.as_tensor(y_valid_flat, dtype=torch.float32, device=device)
    else:
        X_valid_t = y_valid_t = None

    if early_stopping:
        if not validation_available:
            raise ValueError("Early stopping requires validation data.")
        best_model_checkpoint = None
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0
    else:
        best_model_checkpoint = None
        best_val_loss = None
        best_epoch = None

    for epoch in range(1, n_epochs + 1):
        # Next epoch for training
        train_loss, train_kl = train_model(
            model, optimizer, X_train_t, y_train_t, training_config, batch_size, device, epoch, n_epochs
        )

        val_loss = log_loss = val_kl = None
        if validation_available:
            val_loss, log_loss, val_kl = validate_model(
                model, X_valid_t, y_valid_t, training_config, batch_size, device, epoch=epoch, num_epochs=n_epochs
            )

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_checkpoint = deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    print(
                        f"Early stopping | Best epoch: {epoch - early_stopping_patience}. Best validation loss: {best_val_loss:.4f}"
                    )
                    break

        val_str = f"{val_loss:.4f}" if val_loss is not None else "Unavailable"
        print(
            "Epoch: {:>4d}, Training Loss {:>12,.4f} Training KL {:>12,.4f}, Validation Loss {:>12}, Validation log {:>12}, Validation KL {:>12}".format(
                epoch, train_loss, train_kl, val_str, f"{log_loss:.4f}" if log_loss is not None else "Unavailable", f"{val_kl:.4f}" if val_kl is not None else "Unavailable"
            )
        )

        # Update learning rate
        lr_scheduler.step()

    if early_stopping:
        assert best_model_checkpoint is not None, "No best model checkpoint has been found"
        print(f"Restoring best model with val_loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_checkpoint)

    model.n_epochs = n_epochs
    model.num_samples = num_samples

    return model, best_epoch, best_val_loss

class BayesianDistNetModel:
    def __init__(
        self,
        n_input_features,
        device,
        training_config=None,
        model_config=None,
        X_valid=None,
        y_valid=None,
        early_stopping=False,
    ):
        self.n_input_features = n_input_features
        self.device = device
        self.training_config = training_config or training_hparams
        self.model_config = model_config or model_hparams
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.early_stopping = early_stopping
        self.model = None
        self.best_epoch = None
        self.best_val_loss = None
        self.n_epochs = self.training_config["n_epochs"]

    def train(self, X_train, y_train):
        model, best_epoch, best_val_loss = train_bayesianDistNet(
            X_train,
            y_train,
            device=self.device,
            X_valid_flat=self.X_valid,
            y_valid_flat=self.y_valid,
            early_stopping=self.early_stopping,
            training_config=self.training_config,
            model_config=self.model_config,
        )
        self.model = model
        self.best_epoch = best_epoch
        self.best_val_loss = best_val_loss

    def predict(self, X_test, use_batching=False):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train() before predict().")

        self.model.eval()
        X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        n_ens = self.training_config["n_ens"]
        batch_size = self.training_config["batch_size"]

        preds = []
        with torch.no_grad():
            if use_batching:
                for start in range(0, X_test_t.size(0), batch_size):
                    xb = X_test_t[start : (start + batch_size)]
                    if xb.numel() == 0:
                        continue

                    outputs = torch.zeros(xb.size(0), n_ens, device=self.device)
                    for j in range(n_ens):
                        net_out, _kl = self.model(xb)
                        outputs[:, j] = net_out.flatten()

                    outputs = outputs + EPSILON
                    mu = torch.exp(torch.log(outputs).mean(dim=1, keepdim=True))
                    sigma = torch.log(outputs).std(dim=1, keepdim=True)
                    preds.append(torch.cat([sigma, mu], dim=1).cpu())
            else:
                if X_test_t.size(0) == 0:
                    return np.empty((0, 2), dtype=np.float32)

                outputs = torch.zeros(X_test_t.size(0), n_ens, device=self.device)
                for j in range(n_ens):
                    net_out, _kl = self.model(X_test_t)
                    outputs[:, j] = net_out.flatten()

                outputs = outputs + EPSILON
                mu = torch.exp(torch.log(outputs).mean(dim=1, keepdim=True))
                sigma = torch.log(outputs).std(dim=1, keepdim=True)
                preds.append(torch.cat([sigma, mu], dim=1).cpu())

        if not preds:
            return np.empty((0, 2), dtype=np.float32)

        return torch.cat(preds, dim=0).numpy()

BayesianDistnetModel = BayesianDistNetModel
