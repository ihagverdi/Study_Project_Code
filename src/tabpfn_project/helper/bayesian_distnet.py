"""
Self-contained Bayesian DistNet replication module.

This file inlines the old-project transitive dependencies that are required for
Bayesian DistNet architecture, training, losses, censoring handling, and
post-hoc distribution metrics.

Key compatibility notes:
- No imports from old_project are used.
- The original computational ordering is preserved where practical.
- All legacy torch.no_grad usages are replaced with torch.inference_mode.
- Filesystem checkpointing is replaced with in-memory snapshots.
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset, TensorDataset


class _DeviceRegistry:
    """Global device registry.

    Inputs:
    - none

    Outputs:
    - attribute device: torch.device used by all legacy functions.

    Semantic role:
    - Mirrors old helper.device module-level mutable device handle.
    """

    def __init__(self) -> None:
        self.device: Optional[torch.device] = None


# Matches old helper.device usage pattern.
device = _DeviceRegistry()


# ---------------------------------------------------------------------------
# Legacy defaults and config constants
# ---------------------------------------------------------------------------

TRAINING_CONFIG_INI_DEFAULTS: Dict[str, object] = {
    "n_epochs": 250,
    "n_expected_epochs": 500,
    "loss_fn": "LOG_NORMAL",
    "start_rate": 1e-3,
    "end_rate": 1e-5,
    "clip_gradient_norm": 1e-2,
    "n_fcdepth": 16,
    "output_size": 2,
    "drop_value": 0.0,
    "preprocess_mode": "NONE",
    "split_ratio": 0.2,
    "early_stop": 20,
    "seed": 0,
    "n_ens": 16,
    "n_optim": "SGD",
    "batch_size": 32,
    "beta_type": 0.1,
    "posterior_mu1": 0.0,
    "posterior_mu2": 0.1,
    "posterior_sigma1": -3.0,
    "posterior_sigma2": 0.1,
    "mixture_pi": 0.5,
    "prior_sigma1": 0.3,
    "prior_sigma2": 0.01,
}

TRAINING_CONFIG_SECTION_OVERRIDES: Dict[str, Dict[str, object]] = {
    "NORMAL": {"loss_fn": "NORMAL", "output_size": 2},
    "LOGNORMAL": {"loss_fn": "LOG_NORMAL", "output_size": 2},
    "INVGAUSS": {"loss_fn": "INVERSE_GAUSSIAN", "output_size": 2},
    "EXPONENTIAL": {"loss_fn": "EXPONENTIAL", "output_size": 1},
    "BAYESIAN_INVGAUSS": {"loss_fn": "BAYESIAN_INVGAUSS", "output_size": 1},
    "BAYESIAN_LOGNORMAL": {"loss_fn": "BAYESIAN_LOGNORMAL", "output_size": 1},
}


# ---------------------------------------------------------------------------
# Loss enums and helpers
# ---------------------------------------------------------------------------


class LossFunctionTypes(IntEnum):
    """Legacy loss-function enum.

    Inputs:
    - none

    Outputs:
    - integer enum values used by training config and dispatcher.

    Semantic role:
    - Provides exact identifiers used in legacy getLossFunction logic.
    """

    MSE = 0
    EXPONENTIAL = 1
    INVERSE_GAUSSIAN = 2
    LOG_NORMAL = 3
    NORMAL = 4
    BAYESIAN_NORMAL = 5
    BAYESIAN_INVGAUSS = 6
    BAYESIAN_LOGNORMAL = 7
    BAYESIAN_EXPONENTIAL = 8


# Constants from legacy loss_functions.py
EPSILON = 1e-10
TWO_PI = 2 * np.pi
LOG_2PI = np.log(TWO_PI)
SQRT_TWO = np.sqrt(2.0)
HALF = 0.5
E = 2.71828182845904523536028


def _resolve_loss_type(loss_value: object) -> LossFunctionTypes:
    """Resolve a loss config value into LossFunctionTypes.

    Inputs:
    - loss_value: LossFunctionTypes or string enum name.

    Outputs:
    - LossFunctionTypes

    Semantic role:
    - Normalizes config loading and wrapper overrides.
    """

    if isinstance(loss_value, LossFunctionTypes):
        return loss_value
    if isinstance(loss_value, str):
        return LossFunctionTypes[loss_value]
    raise ValueError(f"Unknown loss value: {loss_value}")


def getNumberOfParameters(loss_function_type: LossFunctionTypes) -> int:
    """Get number of model outputs for a selected loss.

    Inputs:
    - loss_function_type: scalar enum selecting distribution.

    Outputs:
    - int output-size.

    Semantic role:
    - Reproduces old output-layer width selection.
    """

    logger = logging.getLogger()
    if loss_function_type == LossFunctionTypes.MSE:
        return 1
    if loss_function_type == LossFunctionTypes.EXPONENTIAL:
        return 1
    if loss_function_type == LossFunctionTypes.INVERSE_GAUSSIAN:
        return 2
    if loss_function_type == LossFunctionTypes.LOG_NORMAL:
        return 2
    if loss_function_type == LossFunctionTypes.NORMAL:
        return 2
    if loss_function_type == LossFunctionTypes.BAYESIAN_EXPONENTIAL:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_NORMAL:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_INVGAUSS:
        return 1
    if loss_function_type == LossFunctionTypes.BAYESIAN_LOGNORMAL:
        return 1

    logger.error("Unknown loss function: %s", loss_function_type)
    sys.exit()


def getLossFunction(loss_function_type: LossFunctionTypes):
    """Get callable loss function.

    Inputs:
    - loss_function_type: scalar enum.

    Outputs:
    - callable taking (prediction, observation, reduce=True).

    Semantic role:
    - Exact dispatcher from legacy training loop.
    """

    logger = logging.getLogger()
    if loss_function_type == LossFunctionTypes.MSE:
        return nn.MSELoss()
    if loss_function_type == LossFunctionTypes.EXPONENTIAL:
        return expo_loss
    if loss_function_type == LossFunctionTypes.INVERSE_GAUSSIAN:
        return invgauss_loss
    if loss_function_type == LossFunctionTypes.LOG_NORMAL:
        return lognormal_loss
    if loss_function_type == LossFunctionTypes.NORMAL:
        return normal_loss
    if loss_function_type == LossFunctionTypes.BAYESIAN_EXPONENTIAL:
        return expo_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_NORMAL:
        return normal_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_LOGNORMAL:
        return lognorm_bayesian
    if loss_function_type == LossFunctionTypes.BAYESIAN_INVGAUSS:
        return invgauss_bayesian

    logger.error("Unknown loss function: %s", loss_function_type)
    sys.exit()


# ---------------------------------------------------------------------------
# Legacy config handler replacements (single-file)
# ---------------------------------------------------------------------------


def getTrainingConfig(
    n_epochs: int = int(TRAINING_CONFIG_INI_DEFAULTS["n_epochs"]),
    n_expected_epochs: int = int(TRAINING_CONFIG_INI_DEFAULTS["n_expected_epochs"]),
    n_optim: str = str(TRAINING_CONFIG_INI_DEFAULTS["n_optim"]),
    batch_size: int = int(TRAINING_CONFIG_INI_DEFAULTS["batch_size"]),
    loss_fn: LossFunctionTypes = LossFunctionTypes[TRAINING_CONFIG_INI_DEFAULTS["loss_fn"]],
    start_rate: float = float(TRAINING_CONFIG_INI_DEFAULTS["start_rate"]),
    end_rate: float = float(TRAINING_CONFIG_INI_DEFAULTS["end_rate"]),
    clip_gradient_norm: Optional[float] = float(TRAINING_CONFIG_INI_DEFAULTS["clip_gradient_norm"]),
    split_ration: float = 0.8,
    seed: int = int(TRAINING_CONFIG_INI_DEFAULTS["seed"]),
    n_ens: int = int(TRAINING_CONFIG_INI_DEFAULTS["n_ens"]),
    beta_type: object = TRAINING_CONFIG_INI_DEFAULTS["beta_type"],
    early_stop: int = int(TRAINING_CONFIG_INI_DEFAULTS["early_stop"]),
) -> Dict[str, object]:
    """Return training configuration dictionary.

    Inputs:
    - Scalars controlling optimization, scheduling, and VI sampling.

    Outputs:
    - dict with training keys consumed by train/train_model/validate_model.

    Semantic role:
    - Preserves old key names, including split_ration typo for fidelity.
    """

    return {
        "n_epochs": n_epochs,
        "n_expected_epochs": n_expected_epochs,
        "n_optim": n_optim,
        "batch_size": batch_size,
        "loss_fn": loss_fn,
        "start_rate": start_rate,
        "end_rate": end_rate,
        "clip_gradient_norm": clip_gradient_norm,
        "split_ration": split_ration,
        "split_ratio": 1.0 - split_ration,
        "seed": seed,
        "n_ens": n_ens,
        "beta_type": beta_type,
        "early_stop": early_stop,
    }


def getModelConfig(
    n_fcdepth: int = int(TRAINING_CONFIG_INI_DEFAULTS["n_fcdepth"]),
    output_size: int = int(TRAINING_CONFIG_INI_DEFAULTS["output_size"]),
    drop_value: float = float(TRAINING_CONFIG_INI_DEFAULTS["drop_value"]),
) -> Dict[str, float]:
    """Return model configuration dictionary.

    Inputs:
    - n_fcdepth: hidden width.
    - output_size: output neuron count.
    - drop_value: dropout probability for deterministic DistNet path.

    Outputs:
    - dict including DistNet and Bayesian prior/posterior parameters.

    Semantic role:
    - Consolidates old getModelConfig + INI-provided Bayesian keys.
    """

    return {
        "n_fcdepth": n_fcdepth,
        "output_size": output_size,
        "drop_value": drop_value,
        "posterior_mu1": float(TRAINING_CONFIG_INI_DEFAULTS["posterior_mu1"]),
        "posterior_mu2": float(TRAINING_CONFIG_INI_DEFAULTS["posterior_mu2"]),
        "posterior_sigma1": float(TRAINING_CONFIG_INI_DEFAULTS["posterior_sigma1"]),
        "posterior_sigma2": float(TRAINING_CONFIG_INI_DEFAULTS["posterior_sigma2"]),
        "mixture_pi": float(TRAINING_CONFIG_INI_DEFAULTS["mixture_pi"]),
        "prior_sigma1": float(TRAINING_CONFIG_INI_DEFAULTS["prior_sigma1"]),
        "prior_sigma2": float(TRAINING_CONFIG_INI_DEFAULTS["prior_sigma2"]),
    }


def trainingConfigToStr(config: Dict[str, object]) -> str:
    """Serialize training config to human-readable text.

    Inputs:
    - config dict from getTrainingConfig/parseConfig.

    Outputs:
    - multiline string.

    Semantic role:
    - Mirrors old training logger payload.
    """

    split_value = config["split_ration"] if "split_ration" in config else config.get("split_ratio", 0.2)
    return (
        "Training Config:\n"
        + "\tNumber of Epochs: {}\n".format(config["n_epochs"])
        + "\tExpected number of Epochs: {}\n".format(config["n_expected_epochs"])
        + "\tOptimizer: {}\n".format(config["n_optim"])
        + "\tBatch size: {}\n".format(config["batch_size"])
        + "\tLoss function: {}\n".format(config["loss_fn"].name)
        + "\tStart rate: {}\n".format(config["start_rate"])
        + "\tEnd rate: {}\n".format(config["end_rate"])
        + "\tGradient Clipping: {}\n".format(config["clip_gradient_norm"])
        + "\tSplit ratio: {}\n".format(split_value)
        + "\tSeed: {}\n".format(config["seed"])
        + "\tSamples for Variational Inference: {}\n".format(config["n_ens"])
        + "\tVariational Inference beta type: {}".format(config["beta_type"])
    )


def modelConfigToStr(config: Dict[str, object]) -> str:
    """Serialize model config to human-readable text.

    Inputs:
    - config dict from getModelConfig/parseConfig.

    Outputs:
    - multiline string.

    Semantic role:
    - Mirrors old model logger payload.
    """

    return (
        "Disnet Config:\n"
        + "\tFirst FC layer size: {}\n".format(config["n_fcdepth"])
        + "\tDropout p = {}\n".format(config["drop_value"])
        + "\tOutput size: {}\n".format(config["output_size"])
        + "\tposterior_mu1: {}".format(config["posterior_mu1"])
        + "\tposterior_mu2: {}".format(config["posterior_mu2"])
        + "\tposterior_sigma1: {}".format(config["posterior_sigma1"])
        + "\tposterior_sigma2: {}".format(config["posterior_sigma2"])
        + "\tmixture pi: {}".format(config["mixture_pi"])
        + "\tprior sigma1: {}".format(config["prior_sigma1"])
        + "\tprior sigma2: {}".format(config["prior_sigma2"])
    )


def parseConfig(config_section: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build configs equivalent to old INI parsing.

    Inputs:
    - config_section: one of DEFAULT, NORMAL, LOGNORMAL, INVGAUSS,
      EXPONENTIAL, BAYESIAN_INVGAUSS, BAYESIAN_LOGNORMAL.

    Outputs:
    - (training_config, model_config)

    Semantic role:
    - In-file replacement for old config/training_config.ini parser.
    """

    logger = logging.getLogger()
    section = str(config_section).upper()
    allowed = {"DEFAULT", *TRAINING_CONFIG_SECTION_OVERRIDES.keys()}
    if section not in allowed:
        logger.error("Training config %s section was not found.", section)
        sys.exit()

    default_cfg = copy.deepcopy(TRAINING_CONFIG_INI_DEFAULTS)
    if section != "DEFAULT":
        for key, value in TRAINING_CONFIG_SECTION_OVERRIDES[section].items():
            default_cfg[key] = value

    training_config = getTrainingConfig(
        n_epochs=int(default_cfg["n_epochs"]),
        n_expected_epochs=int(default_cfg["n_expected_epochs"]),
        n_optim=str(default_cfg["n_optim"]),
        batch_size=int(default_cfg["batch_size"]),
        loss_fn=LossFunctionTypes[str(default_cfg["loss_fn"])],
        start_rate=float(default_cfg["start_rate"]),
        end_rate=float(default_cfg["end_rate"]),
        clip_gradient_norm=float(default_cfg["clip_gradient_norm"]),
        split_ration=1.0 - float(default_cfg["split_ratio"]),
        seed=int(default_cfg["seed"]),
        n_ens=int(default_cfg["n_ens"]),
        beta_type=default_cfg["beta_type"],
        early_stop=int(default_cfg["early_stop"]),
    )
    training_config["split_ratio"] = float(default_cfg["split_ratio"])

    model_config = getModelConfig(
        n_fcdepth=int(default_cfg["n_fcdepth"]),
        output_size=int(default_cfg["output_size"]),
        drop_value=float(default_cfg["drop_value"]),
    )

    return training_config, model_config


# ---------------------------------------------------------------------------
# Legacy metrics utilities
# ---------------------------------------------------------------------------


def calculate_kl(mu_p: torch.Tensor, sig_p: torch.Tensor, mu_q: torch.Tensor, sig_q: torch.Tensor) -> torch.Tensor:
    """Compute elementwise Gaussian KL and reduce to scalar sum.

    Inputs:
    - mu_p, sig_p: tensors of shape [...], prior moments.
    - mu_q, sig_q: tensors of shape [...], posterior moments.

    Outputs:
    - scalar tensor KL divergence sum.

    Semantic role:
    - Preserved helper from old metrics.py.
    """

    kl = 0.5 * (
        2 * torch.log(sig_p / sig_q)
        - 1
        + (sig_q / sig_p).pow(2)
        + ((mu_p - mu_q) / sig_p).pow(2)
    ).sum()
    return kl


def get_beta(batch_idx: int, m: int, beta_type: object, epoch: int, num_epochs: int) -> float:
    """Compute VI beta schedule value.

    Inputs:
    - batch_idx: current mini-batch index.
    - m: number of batches.
    - beta_type: float or schedule string.
    - epoch: current epoch index.
    - num_epochs: total epochs.

    Outputs:
    - scalar beta coefficient.

    Semantic role:
    - Preserved helper from old metrics.py.
    """

    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - batch_idx) / (2**m - 1)
    elif beta_type == "Soenderby":
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta


# ---------------------------------------------------------------------------
# Legacy layer wrappers
# ---------------------------------------------------------------------------


class ModuleWrapper(nn.Module):
    """Wrapper with universal forward pass and KL aggregation.

    Inputs:
    - x: tensor [B, D_in] or compatible shape consumed by children.

    Outputs:
    - tuple (y, kl):
      y is child-module forward output tensor.
      kl is scalar sum of child kl_loss() where available.

    Semantic role:
    - Reproduces old layers.misc.ModuleWrapper behavior.
    """

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name: str, value: object) -> None:
        """Propagate arbitrary flag to children that support set_flag.

        Inputs:
        - flag_name: attribute name.
        - value: attribute value.

        Outputs:
        - none

        Semantic role:
        - Legacy compatibility utility for wrapper hierarchy.
        """

        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, "set_flag"):
                m.set_flag(flag_name, value)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all children in order and accumulate KL.

        Inputs:
        - x: tensor [B, D].

        Outputs:
        - y: tensor [B, D_out]
        - kl: scalar tensor

        Semantic role:
        - Exact old forward contract used by BayesDistNetFCN.
        """

        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, "kl_loss"):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):
    """Flatten helper layer.

    Inputs:
    - x: tensor [B, ...]

    Outputs:
    - tensor [B, num_features]

    Semantic role:
    - Legacy utility retained for completeness.
    """

    def __init__(self, num_features: int):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten input to [B, num_features].

        Inputs:
        - x: tensor [B, ...]

        Outputs:
        - flattened tensor [B, num_features]
        """

        return x.view(-1, self.num_features)


class BBBLinear2(ModuleWrapper):
    """Bayesian linear layer with local reparameterization.

    Inputs:
    - x: tensor [B, in_features]

    Outputs:
    - tensor [B, out_features]

    Semantic role:
    - Replicates old layers.BBBLinear2 exactly for Bayesian DistNet.
    """

    def __init__(self, in_features: int, out_features: int, model_config: Dict[str, float], bias: bool = False, name: str = "BBBLinear"):
        super(BBBLinear2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name

        self.prior_mu = 0.0
        self.prior_sigma = 0.1
        self.prior_sigma1 = model_config["prior_sigma1"]
        self.prior_sigma2 = model_config["prior_sigma2"]
        self.pi = model_config["mixture_pi"]

        self.W_mu = Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features))

        if self.use_bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_rho = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.posterior_mu1 = model_config["posterior_mu1"]
        self.posterior_mu2 = model_config["posterior_mu2"]
        self.posterior_sigma1 = model_config["posterior_sigma1"]
        self.posterior_sigma2 = model_config["posterior_sigma2"]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize posterior parameters.

        Inputs:
        - none

        Outputs:
        - none

        Semantic role:
        - Exact old initialization (normal with configurable means/stds).
        """

        self.W_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
        self.W_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

        if self.use_bias:
            self.bias_mu.data.normal_(self.posterior_mu1, self.posterior_mu2)
            self.bias_rho.data.normal_(self.posterior_sigma1, self.posterior_sigma2)

    def gaussian_prior(self, x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
        """Evaluate Gaussian density for prior components.

        Inputs:
        - x: tensor [...]
        - mu: scalar mean
        - sigma: scalar std

        Outputs:
        - tensor [...]

        Semantic role:
        - Prior mixture density term for KL.
        """

        PI = 3.1415926535897
        scaling = 1.0 / np.sqrt(2.0 * PI * (sigma**2))
        bell = torch.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
        return scaling * bell

    def gaussian(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian density for variational posterior.

        Inputs:
        - x, mu, sigma: tensors with broadcast-compatible shapes.

        Outputs:
        - tensor of same broadcasted shape.

        Semantic role:
        - Variational log-density term for KL.
        """

        PI = 3.1415926535897
        scaling = 1.0 / torch.sqrt(2.0 * PI * (sigma**2))
        bell = torch.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
        return scaling * bell

    def log_prior_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log prior density sum.

        Inputs:
        - x: sampled weight tensor [out_features, in_features].

        Outputs:
        - scalar tensor sum log p(x).

        Semantic role:
        - First KL component for Bayesian layer.
        """

        first_gaussian = self.pi * self.gaussian_prior(x, self.prior_mu, self.prior_sigma1)
        second_gaussian = (1 - self.pi) * self.gaussian_prior(x, self.prior_mu, self.prior_sigma2)
        return torch.log(first_gaussian + second_gaussian).sum()

    def log_variational_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Compute variational posterior log-density sum.

        Inputs:
        - x: sampled weight tensor [out_features, in_features].

        Outputs:
        - scalar tensor sum log q(x).

        Semantic role:
        - Second KL component for Bayesian layer.
        """

        return torch.log(self.gaussian(x, self.W_mu, self.W_sigma)).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample activations using local reparameterization.

        Inputs:
        - x: tensor [B, in_features]

        Outputs:
        - tensor [B, out_features]

        Semantic role:
        - Preserves old stochastic forward path and numerical ordering.
        """

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma**2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x**2, self.W_sigma**2, bias_var)
        act_std = torch.sqrt(act_var)

        eps = torch.empty(act_mu.size()).normal_(0, 1).to(device.device)
        return act_mu + act_std * eps

    def kl_loss(self) -> torch.Tensor:
        """Sample weights and compute layer KL contribution.

        Inputs:
        - none

        Outputs:
        - scalar tensor KL(q||p)

        Semantic role:
        - Per-layer KL term used by ModuleWrapper.forward.
        """

        W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(device.device)
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        self.weights = self.W_mu + W_eps * self.W_sigma
        kl = self.log_variational_sum(self.weights) - self.log_prior_sum(self.weights)
        return kl


# ---------------------------------------------------------------------------
# Legacy model definitions
# ---------------------------------------------------------------------------

class DistNetFCN(nn.Module):
    pass


class BayesDistNetFCN(ModuleWrapper):
    """Legacy Bayesian DistNet model with BBBLinear2 layers.

    Inputs:
    - x: tensor [B, num_features]

    Outputs:
    - tuple (y, kl):
      y tensor [B, output_size]
      kl scalar aggregated over Bayesian layers.

    Semantic role:
    - Core architecture to replicate old Bayesian DistNet behavior.
    """

    def __init__(self, num_features: int, config: Dict[str, object] = getModelConfig(), name_suffix: str = ""):
        super().__init__()
        self.name_suffix = name_suffix

        self.n_fcdepth = config["n_fcdepth"]
        self.output_size = config["output_size"]
        self.drop_value = config["drop_value"]

        in_channel = num_features
        out_channel = self.n_fcdepth
        self.layer1 = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name="fc1"),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        in_channel = out_channel
        out_channel = self.n_fcdepth
        self.layer2 = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name="fc2"),
            nn.BatchNorm1d(out_channel),
            nn.Softplus(),
        )

        in_channel = out_channel
        out_channel = self.output_size
        self.layer_end = nn.Sequential(
            BBBLinear2(in_channel, out_channel, config, name="fc3"),
            nn.Softplus(),
        )

    def toStr(self) -> str:
        """Return legacy model identifier string.

        Inputs:
        - none

        Outputs:
        - string name.
        """

        return "Bayes_Distnet" + "_" + self.name_suffix


# ---------------------------------------------------------------------------
# Legacy model utilities
# ---------------------------------------------------------------------------


def getModel(net_type: str, model_config: Dict[str, object], num_features: int) -> nn.Module:
    """Instantiate model by legacy net_type string.

    Inputs:
    - net_type: distnet or bayes_distnet.
    - model_config: model configuration dict.
    - num_features: input feature count.

    Outputs:
    - torch.nn.Module instance.

    Semantic role:
    - Mirrors old model_util.getModel dispatcher.
    """

    if net_type == "distnet":
        return DistNetFCN(num_features, model_config)
    if net_type == "bayes_distnet":
        return BayesDistNetFCN(num_features, model_config)
    raise ValueError("Unknown net type.")


def weights_init(m: nn.Module) -> None:
    """Apply legacy Xavier initialization for Linear/Conv2d modules.

    Inputs:
    - m: module visited by model.apply.

    Outputs:
    - none

    Semantic role:
    - Preserves old initialization scheme.
    """

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        m.bias.data.zero_()
        m.bias.data.fill_(0.01)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters.

    Inputs:
    - model: nn.Module

    Outputs:
    - integer number of trainable parameters.

    Semantic role:
    - Legacy logging helper.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Legacy loss implementations
# ---------------------------------------------------------------------------


def _approx_erf(x: torch.Tensor) -> torch.Tensor:
    """Approximate error function using Abramowitz-Stegun polynomial.

    Inputs:
    - x: tensor [N, ...]

    Outputs:
    - erf(x) approximation tensor with same shape.

    Semantic role:
    - Legacy approximation used in CDF computations.
    """

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    neg_mask = x < 0
    torch.abs_(x)

    t = 1 / (1 + (p * x))
    polynomial = (a1 * t) + (a2 * (t**2)) + (a3 * (t**3)) + (a4 * (t**4)) + (a5 * (t**5))
    erf = 1 - (polynomial * torch.exp(-(x**2)))
    erf[neg_mask] *= -1
    return erf


def _standard_gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """Approximate standard Gaussian CDF.

    Inputs:
    - x: tensor [N, ...]

    Outputs:
    - cdf(x) tensor with same shape.

    Semantic role:
    - Legacy helper for Normal/Lognormal survival terms.
    """

    return HALF * (1 + _approx_erf(x / SQRT_TWO))


def expo_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Exponential NLL with censoring support.

    Inputs:
    - prediction: tensor [B, 1], positive scale proxy.
    - observation: tensor [B, 2], columns [target, sol_flag].
    - reduce: if True returns scalar mean NLL, else returns [B] NLL.

    Outputs:
    - scalar tensor or vector tensor [B].

    Semantic role:
    - Legacy negative log-likelihood objective.
    """

    scale = prediction[:, 0:1] + EPSILON
    scale = 1.0 / scale
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    scale_1 = scale[sol_flag]
    scale_2 = scale[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    llh[sol_flag] = torch.flatten(torch.log(scale_1) - (scale_1 * target_1))

    cdf = 1 - torch.exp(-scale_2 * target_2)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def expo_scipy_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Scipy exponential NLL reference implementation.

    Inputs:
    - prediction: tensor [B, 1]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [1, B] tensor matching legacy behavior.

    Semantic role:
    - Legacy validation helper using scipy.stats.expon.
    """

    with torch.inference_mode():
        scale = prediction[:, 0:1] + EPSILON
        scale = 1.0 / scale
        target = observation[:, 0] + EPSILON

        nll = []
        for s, t in zip(scale, target):
            nll.append(stats.expon.logpdf(t.item(), loc=0, scale=1.0 / s.item()))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def lognormal_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Lognormal NLL with censoring support.

    Inputs:
    - prediction: tensor [B, 2], columns [sigma, mu_scale].
    - observation: tensor [B, 2], columns [target, sol_flag].
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] tensor of NLL.

    Semantic role:
    - Legacy DistNet/likelihood objective for LOG_NORMAL.
    """

    sigma = prediction[:, 0:1] + EPSILON
    mu = prediction[:, 1:2] + EPSILON
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help1 = HALF * ((torch.log(target_1) - torch.log(mu_1)) / sigma_1) ** 2
    llh[sol_flag] = torch.flatten(-torch.log(target_1) - torch.log(sigma_1) - pdf_help1)

    cdf_help1 = (torch.log(target_2) - torch.log(mu_2)) / (SQRT_TWO * sigma_2)
    cdf = HALF + (HALF * _approx_erf(cdf_help1))
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def lognormal_scipy_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Scipy lognormal NLL reference implementation.

    Inputs:
    - prediction: tensor [B, 2]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [1, B] tensor.

    Semantic role:
    - Legacy validation helper using scipy.stats.lognorm.
    """

    with torch.inference_mode():
        mu = prediction[:, 1] + EPSILON
        sigma = prediction[:, 0] + EPSILON
        target = observation[:, 0] + EPSILON

        nll = []
        for m, s, t in zip(mu, sigma, target):
            nll.append(stats.lognorm.logpdf(t.item(), s.item(), loc=0, scale=m.item()))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def _pdf_invgauss(x: torch.Tensor, mu: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    """Inverse-Gaussian PDF helper.

    Inputs:
    - x: tensor [N] or [B, N]
    - mu: tensor broadcastable to x
    - lambda_: tensor broadcastable to x

    Outputs:
    - pdf tensor with broadcasted shape.

    Semantic role:
    - Used in inverse-Gaussian numerical CDF integration.
    """

    helper = -(lambda_ * (x - mu) ** 2) / (2 * x * (mu**2))
    return (torch.sqrt(lambda_) / torch.sqrt(TWO_PI * (x**3))) * torch.exp(helper)


def invgauss_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Inverse-Gaussian NLL with censoring support.

    Inputs:
    - prediction: tensor [B, 2], columns [mu, lambda].
    - observation: tensor [B, 2], columns [target, sol_flag].
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] tensor of NLL.

    Semantic role:
    - Legacy objective with trapezoidal CDF approximation.
    """

    mu = prediction[:, 0] + EPSILON
    lambda_ = prediction[:, 1] + EPSILON
    target = observation[:, 0] + EPSILON
    sol_flag = observation[:, 1] == 1

    lambda_1 = lambda_[sol_flag]
    lambda_2 = lambda_[~sol_flag].unsqueeze(-1)
    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag].unsqueeze(-1)
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag].unsqueeze(-1)

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help1 = (lambda_1 * (target_1 - mu_1) ** 2) / (2 * target_1 * (mu_1**2))
    llh[sol_flag] = torch.flatten((HALF * torch.log(lambda_1)) - (3.0 * HALF * torch.log(target_1)) - pdf_help1)

    STEPS = 100
    if target_2.nelement() != 0:
        xs = torch.stack([torch.arange(1, STEPS + 1) / float(STEPS) * i.item() for i in target_2]).to(device.device)
        pdfs = _pdf_invgauss(xs, mu_2, lambda_2)
        cdfs = torch.clamp(torch.clamp(torch.trapz(pdfs, xs), max=1), min=0)
        llh[~sol_flag] = torch.flatten(torch.log(1 - cdfs + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def invgauss_scipy_loss(prediction: torch.Tensor, target: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Scipy inverse-Gaussian NLL reference implementation.

    Inputs:
    - prediction: tensor [B, 2]
    - target: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [1, B] tensor.

    Semantic role:
    - Legacy validation helper using scipy.stats.invgauss.
    """

    with torch.inference_mode():
        mu = prediction[:, 0] + EPSILON
        lambda_ = prediction[:, 1] + EPSILON
        target = target[:, 0] + EPSILON

        nll = []
        for m, s, t in zip(mu, lambda_, target):
            nll.append(stats.invgauss.logpdf(t.item(), m.item() / s.item(), loc=0, scale=s.item()))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


def _pdf_normal(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Normal PDF helper.

    Inputs:
    - x, mu, sigma tensors broadcastable together.

    Outputs:
    - tensor PDF values.

    Semantic role:
    - Kept for legacy completeness.
    """

    helper = HALF * ((x - mu) / sigma) ** 2
    return 1.0 / (sigma * np.sqrt(TWO_PI)) * torch.exp(-helper)


def normal_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Normal NLL with censoring support.

    Inputs:
    - prediction: tensor [B, 2], columns [mu, sigma].
    - observation: tensor [B, 2], columns [target, sol_flag].
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] tensor of NLL.

    Semantic role:
    - Legacy objective for NORMAL loss type.
    """

    mu = prediction[:, 0:1]
    sigma = prediction[:, 1:2] + EPSILON
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]

    llh = torch.zeros([prediction.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help = HALF * ((target_1 - mu_1) / sigma_1) ** 2
    llh[sol_flag] = torch.flatten(-torch.log(sigma_1) - pdf_help)

    cdf = _standard_gaussian_cdf((target_2 - mu_2) / sigma_2)
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1.0 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def expo_bayesian(outputs: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Bayesian exponential NLL using ensemble outputs.

    Inputs:
    - outputs: tensor [B, N_ens] sampled runtime predictions.
    - observation: tensor [B, 2], columns [target, sol_flag].
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] NLL tensor.

    Semantic role:
    - Legacy Bayesian likelihood objective.
    """

    scale = torch.reciprocal(outputs.mean(dim=1, keepdim=True))
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    scale_1 = scale[sol_flag]
    scale_2 = scale[~sol_flag]

    llh = torch.zeros([scale.shape[0]], dtype=torch.float32).to(device.device)

    llh[sol_flag] = torch.flatten(torch.log(scale_1) - (scale_1 * target_1))

    cdf = 1 - torch.exp(-scale_2 * target_2)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def normal_bayesian(outputs: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Bayesian normal NLL using ensemble outputs.

    Inputs:
    - outputs: tensor [B, N_ens]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] NLL tensor.

    Semantic role:
    - Legacy Bayesian normal objective.
    """

    outputs += 1e-10
    mu = outputs.mean(dim=1, keepdim=True)
    sigma = outputs.std(dim=1, keepdim=True)
    target = observation[:, 0:1]
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]

    llh = torch.zeros([mu.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help = HALF * ((target_1 - mu_1) / sigma_1) ** 2
    llh[sol_flag] = torch.flatten(-torch.log(sigma_1) - pdf_help)

    cdf = _standard_gaussian_cdf((target_2 - mu_2) / sigma_2)
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1.0 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def lognorm_bayesian(outputs: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Bayesian lognormal NLL using ensemble outputs.

    Inputs:
    - outputs: tensor [B, N_ens]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] NLL tensor over filtered rows.

    Semantic role:
    - Legacy Bayesian lognormal objective.
    """

    flag = outputs.sum(dim=1) > 1e-8
    outputs = outputs[flag]
    observation = observation[flag]
    outputs += 1e-10
    mu = torch.exp(torch.log(outputs).mean(dim=1, keepdim=True))
    sigma = torch.log(outputs).std(dim=1, keepdim=True)
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    sigma_1 = sigma[sol_flag]
    sigma_2 = sigma[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag]

    llh = torch.zeros([outputs.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help1 = HALF * ((torch.log(target_1) - torch.log(mu_1)) / sigma_1) ** 2
    llh[sol_flag] = torch.flatten(-torch.log(target_1) - torch.log(sigma_1) - pdf_help1)

    cdf_help1 = (torch.log(target_2) - torch.log(mu_2)) / (SQRT_TWO * sigma_2)
    cdf = HALF + (HALF * _approx_erf(cdf_help1))
    cdf = torch.clamp(torch.clamp(cdf, max=1), min=0)
    llh[~sol_flag] = torch.flatten(torch.log(1 - cdf + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def invgauss_bayesian(outputs: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Bayesian inverse-Gaussian NLL using ensemble outputs.

    Inputs:
    - outputs: tensor [B, N_ens]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [B] NLL tensor over filtered rows.

    Semantic role:
    - Legacy Bayesian inverse-Gaussian objective.
    """

    flag = outputs.sum(dim=1) > 1e-8
    outputs = outputs[flag]
    observation = observation[flag]
    outputs += 1e-10

    mu = outputs.mean(dim=1, keepdim=True)
    temp_lambda = (1.0 / outputs) - (1.0 / mu)
    lambda_ = outputs.shape[1] / (temp_lambda.sum(dim=1, keepdim=True))
    target = observation[:, 0:1] + EPSILON
    sol_flag = observation[:, 1] == 1

    lambda_1 = lambda_[sol_flag]
    lambda_2 = lambda_[~sol_flag]
    mu_1 = mu[sol_flag]
    mu_2 = mu[~sol_flag]
    target_1 = target[sol_flag]
    target_2 = target[~sol_flag].unsqueeze(-1)

    llh = torch.zeros([outputs.shape[0]], dtype=torch.float32).to(device.device)

    pdf_help1 = (lambda_1 * (target_1 - mu_1) ** 2) / (2 * target_1 * (mu_1**2))
    llh[sol_flag] = torch.flatten((HALF * torch.log(lambda_1)) - (3.0 * HALF * torch.log(target_1)) - pdf_help1)

    STEPS = 100
    if target_2.nelement() != 0:
        xs = torch.stack([torch.arange(1, STEPS + 1) / float(STEPS) * i.item() for i in target_2]).to(device.device)
        pdfs = _pdf_invgauss(xs, mu_2, lambda_2)
        cdfs = torch.clamp(torch.clamp(torch.trapz(pdfs, xs), max=1), min=0)
        llh[~sol_flag] = torch.flatten(torch.log(1 - cdfs + EPSILON))

    return torch.mean(-llh) if reduce else -llh


def normal_scipy_loss(prediction: torch.Tensor, observation: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Scipy normal NLL reference implementation.

    Inputs:
    - prediction: tensor [B, 2]
    - observation: tensor [B, 2]
    - reduce: reduction toggle.

    Outputs:
    - scalar or [1, B] tensor.

    Semantic role:
    - Legacy validation helper using scipy.stats.norm.
    """

    with torch.inference_mode():
        mu = prediction[:, 0] + EPSILON
        sigma = prediction[:, 1] + EPSILON
        target = observation[:, 0] + EPSILON

        nll = []
        for m, s, t in zip(mu, sigma, target):
            nll.append(stats.norm.logpdf(t.item(), loc=m.item(), scale=s.item()))

    llh = torch.tensor([nll], dtype=torch.float32, requires_grad=True)
    return torch.mean(-llh) if reduce else -llh


# ---------------------------------------------------------------------------
# Legacy preprocessing and censoring utilities
# ---------------------------------------------------------------------------


def remove_timeouts(
    runningtimes: np.ndarray,
    cutoff: float,
    features: Optional[np.ndarray] = None,
    sat_ls: Optional[Sequence[str]] = None,
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove instances with any run >= cutoff.

    Inputs:
    - runningtimes: array [N, R]
    - cutoff: scalar timeout threshold.
    - features: array [N, D]
    - sat_ls: list length N

    Outputs:
    - filtered (runningtimes, features, sat_ls)

    Semantic role:
    - Preserved data filtering helper from legacy preprocess module.
    """

    if features is None:
        features = np.array([0] * runningtimes.shape[0])
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance >= cutoff):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    if log:
        print(
            "Discarding {:d} ({:d}) instances because not stated TIMEOUTS".format(
                len(features) - len(new_ft), len(features)
            )
        )
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_instances_with_status(
    runningtimes: np.ndarray,
    features: np.ndarray,
    sat_ls: Optional[Sequence[str]] = None,
    status: str = "CRASHED",
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove instances whose status string contains a target marker.

    Inputs:
    - runningtimes: array [N, R]
    - features: array [N, D]
    - sat_ls: status list length N
    - status: string marker to remove

    Outputs:
    - filtered (runningtimes, features, sat_ls)

    Semantic role:
    - Preserved legacy status-based filter.
    """

    if sat_ls is None:
        print("Could not remove {} instances".format(status))
        sat_ls = [""] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if status not in s:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    if log:
        print(
            "Discarding {:d} ({:d}) instances because of {}".format(
                len(features) - len(new_ft), len(features), status
            )
        )
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_constant_instances(
    runningtimes: np.ndarray,
    features: np.ndarray,
    sat_ls: Optional[Sequence[str]] = None,
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove rows with constant feature vector.

    Inputs:
    - runningtimes: array [N, R]
    - features: array [N, D]
    - sat_ls: optional status list

    Outputs:
    - filtered arrays/lists.

    Semantic role:
    - Preserved legacy feature-constancy filter.
    """

    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if np.std(f) > 0:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    if log:
        print(
            "Discarding {:d} ({:d}) instances because of constant features".format(
                len(features) - len(new_ft), len(features)
            )
        )
    return np.array(new_rt), np.array(new_ft), new_sl


def feature_imputation(features: np.ndarray, impute_val: float = -512, impute_with: str = "median") -> np.ndarray:
    """Impute feature sentinel values.

    Inputs:
    - features: array [N, D]
    - impute_val: sentinel to replace
    - impute_with: strategy name (median supported)

    Outputs:
    - imputed array [N, D]

    Semantic role:
    - Preserved legacy feature-imputation behavior.
    """

    print(features.shape)
    if impute_with == "median":
        for col in range(features.shape[1]):
            med = np.median(features[:, col])
            features[:, col] = [med if i == impute_val else i for i in features[:, col]]
    return features


def remove_zeros(
    runningtimes: np.ndarray,
    features: Optional[np.ndarray] = None,
    sat_ls: Optional[Sequence[str]] = None,
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove instances with non-positive runtimes.

    Inputs:
    - runningtimes: array [N, R]
    - features: array [N, D]
    - sat_ls: list length N

    Outputs:
    - filtered arrays/lists.

    Semantic role:
    - Preserved legacy runtime validity filter.
    """

    if features is None:
        features = np.array([0] * runningtimes.shape[0])
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance <= 0):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    if log:
        print("Discarding {:d} ({:d}) instances because of ZEROS".format(len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def det_constant_features(X: np.ndarray, log: bool = False) -> Tuple[np.ndarray]:
    """Detect constant-feature column indices.

    Inputs:
    - X: array [N, D]

    Outputs:
    - tuple of arrays from np.where over constant columns.

    Semantic role:
    - Preserved legacy constant-feature detector.
    """

    max_ = X.max(axis=0)
    min_ = X.min(axis=0)
    diff = max_ - min_

    det_idx = np.where(diff <= 10e-10)
    if log:
        print("Discarding {:d} ({:d}) features".format(det_idx[0].shape[0], X.shape[1]))
    return det_idx


def det_transformation(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min and max-minus-min vectors.

    Inputs:
    - X: array [N, D]

    Outputs:
    - (min_, max_) vectors shape [D]

    Semantic role:
    - Preserved legacy scaling helper.
    """

    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0) - min_
    return min_, max_


def preprocess_features(
    tra_X: np.ndarray,
    val_X: np.ndarray,
    test_X: Optional[np.ndarray] = None,
    scal: str = "meanstd",
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess feature matrices with train-derived statistics.

    Inputs:
    - tra_X: train array [N_tr, D]
    - val_X: validation array [N_val, D]
    - test_X: optional test array [N_te, D]
    - scal: minmax or meanstd

    Outputs:
    - (tra_X_scaled, val_X_scaled, test_X_scaled)

    Semantic role:
    - Preserved legacy preprocessing used before training.
    """

    if test_X is None:
        test_X = np.array([])

    del_idx = det_constant_features(tra_X, log)
    tra_X = np.delete(tra_X, del_idx, axis=1)
    val_X = np.delete(val_X, del_idx, axis=1)
    if len(test_X) > 0:
        test_X = np.delete(test_X, del_idx, axis=1)

    if scal == "minmax":
        min_, max_ = det_transformation(tra_X)
        tra_X = (tra_X - min_) / max_
        val_X = (val_X - min_) / max_
        if len(test_X) > 0:
            test_X = (test_X - min_) / max_
    else:
        mean_ = tra_X.mean(axis=0)
        std_ = tra_X.std(axis=0)
        tra_X = (tra_X - mean_) / std_
        val_X = (val_X - mean_) / std_
        if len(test_X) > 0:
            test_X = (test_X - mean_) / std_

    return tra_X, val_X, test_X


def preprocess(
    features: np.ndarray,
    runtimes: np.ndarray,
    train_idx: np.ndarray,
    validate_idx: np.ndarray,
    num_train_samples: int,
    lb: int,
    seed: int,
    log: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy preprocessing pipeline including censoring.

    Inputs:
    - features: [N_instances, D]
    - runtimes: [N_instances, 100]
    - train_idx: indices for train instances
    - validate_idx: indices for val instances
    - num_train_samples: context samples per instance used for train flattening
    - lb: censoring lower-bound percent (0 disables censoring)
    - seed: sampling seed for subset selection

    Outputs:
    - X_trn_flat: [N_train_flat, D]
    - X_vld_flat: [N_val_flat, D]
    - y_trn_flat: [N_train_flat, 2] with [scaled_runtime, sol_flag]
    - y_vld_flat: [N_val_flat, 2] with [scaled_runtime, sol_flag]
    - y_train: [N_train_instances, num_train_samples] scaled
    - y_valid: [N_val_instances, 100] scaled

    Semantic role:
    - Preserves old censoring implementation and default lb=0 no-effect path.
    """

    logger = None
    if logging:
        logger = logging.getLogger()

    X_train = features[train_idx, :]
    X_valid = features[validate_idx, :]
    y_train = runtimes[train_idx]
    y_valid = runtimes[validate_idx]

    X_train, X_valid, _ = preprocess_features(X_train, X_valid, scal="meanstd")

    X_trn_flat = np.concatenate([[x for _ in range(100)] for x in X_train])
    X_vld_flat = np.concatenate([[x for _ in range(100)] for x in X_valid])
    y_trn_flat = y_train.flatten().reshape([-1, 1])
    y_vld_flat = y_valid.flatten().reshape([-1, 1])

    subset_idx = list(range(100))
    if num_train_samples != 100:
        if logging and logger is not None:
            logger.info("Cut data down to %d samples with seed %d", num_train_samples, seed)
        rs = np.random.RandomState(seed)
        rs.shuffle(subset_idx)
        subset_idx = subset_idx[:num_train_samples]
        if logger is not None:
            logger.info(subset_idx)

        X_trn_flat = np.concatenate([[x for _ in range(num_train_samples)] for x in X_train])
        y_train = y_train[:, subset_idx]
        y_trn_flat = y_train.flatten().reshape([-1, 1])

    y_max_ = np.max(y_trn_flat)
    y_min_ = 0
    y_trn_flat = (y_trn_flat - y_min_) / y_max_
    y_vld_flat = (y_vld_flat - y_min_) / y_max_
    y_train = (y_train - y_min_) / y_max_
    y_valid = (y_valid - y_min_) / y_max_

    print(np.mean(y_train))

    y_trn_flat = np.c_[y_trn_flat, np.ones(len(y_trn_flat))]
    y_vld_flat = np.c_[y_vld_flat, np.ones(len(y_vld_flat))]

    if logger is not None:
        logger.info("Using lb: %s", lb)

    if lb != 0:
        y_trn_flat = y_train.flatten()
        ytemp = copy.deepcopy(y_trn_flat)
        ytemp.sort(axis=0)
        censored_time = ytemp[int(num_train_samples * (1 - (lb / 100))) - 1]
        idx = int(len(ytemp) * (1 - (lb / 100)) - 1)
        censored_time = ytemp[idx]
        mask = y_trn_flat > censored_time
        y_trn_flat = np.where(~mask, y_trn_flat, censored_time)
        y_trn_flat = np.c_[y_trn_flat, np.where(~mask, 1, 0).flatten()]

    return X_trn_flat, X_vld_flat, y_trn_flat, y_vld_flat, y_train, y_valid


class CustomDataset(Dataset):
    """Legacy GPU-resident dataset wrapper.

    Inputs:
    - features: [N_instances, D]
    - runtimes: [N_instances, 100]
    - train_idx, validate_idx: index arrays
    - num_train_samples: int samples per train instance
    - seed: sampling seed
    - lb: censoring level
    - device_num: cuda index

    Outputs:
    - dataset returning (feature, observation) where observation=[runtime, sol_flag]

    Semantic role:
    - Preserved old data handler structure.
    """

    def __init__(
        self,
        features: np.ndarray,
        runtimes: np.ndarray,
        train_idx: np.ndarray,
        validate_idx: np.ndarray,
        num_train_samples: int,
        seed: int,
        lb: int,
        device_num: int,
    ):
        X_trn_flat, X_vld_flat, y_trn_flat, y_vld_flat, _, _ = preprocess(
            features,
            runtimes,
            train_idx,
            validate_idx,
            num_train_samples,
            lb,
            seed,
            True,
        )

        _device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        self.features = torch.tensor(np.concatenate((X_trn_flat, X_vld_flat)), dtype=torch.float).to(_device)
        self.observations = torch.tensor(np.concatenate((y_trn_flat, y_vld_flat)), dtype=torch.float).to(_device)
        self.train_idx = range(len(X_trn_flat))
        self.validate_idx = range(len(X_trn_flat), len(X_trn_flat) + len(X_vld_flat))

    def getTrainValidSubsetSampler(self):
        """Return train/validation index ranges.

        Inputs:
        - none

        Outputs:
        - tuple(train_idx_range, valid_idx_range)

        Semantic role:
        - Simplified in-file replacement of old sampler utility.
        """

        return self.train_idx, self.validate_idx

    def getNumFeatures(self) -> int:
        """Get feature dimension.

        Inputs:
        - none

        Outputs:
        - integer feature count.
        """

        return self.features.shape[1]

    def __len__(self) -> int:
        """Return dataset row count.

        Inputs:
        - none

        Outputs:
        - integer number of rows.
        """

        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index dataset row.

        Inputs:
        - idx: integer row index.

        Outputs:
        - (feature[D], observation[2]).
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.observations[idx]


# ---------------------------------------------------------------------------
# Legacy export helpers rewritten for in-memory operation
# ---------------------------------------------------------------------------


def dump_res(save_path: Optional[str], data: np.ndarray) -> np.ndarray:
    """In-memory replacement for legacy pickle dump.

    Inputs:
    - save_path: unused string path (kept for signature compatibility)
    - data: numpy array to persist

    Outputs:
    - same numpy array

    Semantic role:
    - Preserves call-site side-effect boundary without filesystem writes.
    """

    logger = logging.getLogger()
    if save_path is not None and save_path != "":
        logger.info("Skipping filesystem dump for %s (memory-only mode)", save_path)
    return data


def exportData(X_data: np.ndarray, data_path: Optional[str], model: nn.Module) -> np.ndarray:
    """Export model outputs using legacy sampling protocol.

    Inputs:
    - X_data: feature array [N, D]
    - data_path: unused output path retained for compatibility
    - model: trained DistNet/BayesDistNet model

    Outputs:
    - numpy array:
      For BayesDistNet: [N, 64*16] sampled runtime outputs.
      For DistNet: [N, output_size] deterministic outputs.

    Semantic role:
    - Reproduces old exportData computation order with memory-only storage.
    """

    outputs: List[object] = []
    model.eval()
    for val_data in X_data:
        val_inputs = torch.tensor(val_data, dtype=torch.float).to(device.device).unsqueeze(0)
        val_inputs = val_inputs.repeat(64, 1).to(device.device)
        if type(model) is BayesDistNetFCN:
            rts: List[float] = []
            for _ in range(16):
                val_pred = model(val_inputs)[0]
                rts = rts + val_pred.flatten().cpu().detach().tolist()
            outputs.append(rts)
        else:
            val_inputs = torch.tensor(val_data, dtype=torch.float).unsqueeze(0)
            val_inputs = val_inputs.to(device.device)
            val_pred = model(val_inputs)
            outputs += val_pred.cpu().detach().tolist()

    arr = np.array(outputs)
    dump_res(data_path, arr)
    return arr


def saveModel(model: nn.Module, model_path: Optional[str]) -> Dict[str, torch.Tensor]:
    """Save model state in-memory.

    Inputs:
    - model: nn.Module
    - model_path: unused string path retained for compatibility

    Outputs:
    - deep-copied state_dict

    Semantic role:
    - Replaces legacy torch.save file output with memory checkpoint.
    """

    logger = logging.getLogger()
    if model_path is not None and model_path != "":
        logger.info("Skipping filesystem model export for %s (memory-only mode)", model_path)
    return copy.deepcopy(model.state_dict())


# ---------------------------------------------------------------------------
# Legacy training loop with memory checkpoints and best-state restore
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    training_config: Dict[str, object],
    total_len: int,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """Run one training epoch.

    Inputs:
    - model: DistNetFCN or BayesDistNetFCN
    - train_loader: yields (inputs[B, D], observations[B, 2])
    - optimizer: torch optimizer
    - training_config: dict with n_ens, loss_fn, clip_gradient_norm, beta_type
    - total_len: unused legacy argument
    - epoch: current epoch index
    - num_epochs: total epoch count

    Outputs:
    - (avg_training_loss, total_kl)

    Semantic role:
    - Exact legacy training-step ordering and Bayesian KL integration.
    """

    _ = total_len
    training_loss = 0.0
    model.train()
    loss_fn = getLossFunction(training_config["loss_fn"])
    n_ens = training_config["n_ens"]
    clip_gradient_norm = training_config["clip_gradient_norm"]
    beta_type = training_config["beta_type"]
    total_kl = 0.0

    n_train_batches = len(train_loader)
    if n_train_batches == 0:
        raise ValueError(
            "train_loader is empty. Ensure at least one training sample after subsampling/splitting."
        )

    for batch_idx, data in enumerate(train_loader):
        inputs, rts = data
        optimizer.zero_grad()

        if type(model) is BayesDistNetFCN:
            kl = 0.0
            batch_size = inputs.shape[0]
            outputs = torch.zeros(batch_size, n_ens).to(device.device)
            for j in range(n_ens):
                net_out, _kl = model(inputs)
                kl += _kl
                outputs[:, j] = net_out.flatten()
            kl /= n_ens

            beta = get_beta(batch_idx, n_train_batches, beta_type, epoch, num_epochs)
            loss = (kl * beta) / n_train_batches + loss_fn(outputs, rts, reduce=False).sum()
            total_kl += (kl * beta) / n_train_batches
        elif type(model) is DistNetFCN:
            outputs = model(inputs)
            loss = loss_fn(outputs, rts)
        else:
            raise ValueError("Unknown net type.")

        loss.backward()

        if clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)
        optimizer.step()

        training_loss += loss.item()

    total_kl_value = total_kl.detach().item() if torch.is_tensor(total_kl) else float(total_kl)
    return training_loss / n_train_batches, total_kl_value


def validate_model(
    model: nn.Module,
    validation_loader: DataLoader,
    training_config: Dict[str, object],
    val_len: int,
    epoch: int = 1,
    num_epochs: int = 1,
) -> Tuple[float, float, float]:
    """Run one validation epoch.

    Inputs:
    - model: DistNetFCN or BayesDistNetFCN
    - validation_loader: yields (inputs[B, D], observations[B, 2])
    - training_config: dict with n_ens, loss_fn, beta_type
    - val_len: unused legacy argument
    - epoch: current epoch index
    - num_epochs: total epoch count

    Outputs:
    - (avg_validation_loss, avg_log_loss, total_kl)

    Semantic role:
    - Legacy validation logic with inference-mode substitution.
    """

    _ = val_len
    loss_fn = getLossFunction(training_config["loss_fn"])
    n_ens = training_config["n_ens"]
    beta_type = training_config["beta_type"]

    model.eval()

    validation_loss = 0.0
    log_loss = 0.0
    total_kl = 0.0
    with torch.inference_mode():
        for batch_idx, val_data in enumerate(validation_loader):
            val_inputs, val_rts = val_data

            if type(model) is BayesDistNetFCN:
                kl = 0.0
                batch_size = val_inputs.shape[0]
                outputs = torch.zeros(batch_size, n_ens).to(device.device)
                for j in range(n_ens):
                    net_out, _kl = model(val_inputs)
                    kl += _kl
                    outputs[:, j] = net_out.flatten()
                kl /= n_ens

                beta = get_beta(batch_idx, len(validation_loader), beta_type, epoch, num_epochs)
                loss = loss_fn(outputs, val_rts, reduce=True).sum()
                val_loss = (kl * beta) / len(validation_loader) + loss
                total_kl += (kl * beta) / len(validation_loader)

            elif type(model) is DistNetFCN:
                outputs = model(val_inputs)
                val_loss = loss_fn(outputs, val_rts)
                loss = val_loss
            else:
                raise ValueError("Unknown net type.")

            validation_loss += val_loss.item()
            log_loss += loss.item()

            if bool(loss != loss):
                torch.set_printoptions(edgeitems=10000)
                logger = logging.getLogger()
                logger.info(log_loss)
                logger.info(outputs)
                logger.info(kl)
                raise RuntimeError("NaN validation loss encountered.")

    total_kl_value = total_kl.detach().item() if torch.is_tensor(total_kl) else float(total_kl)
    return validation_loss / len(validation_loader), log_loss / len(validation_loader), total_kl_value


def train(
    num_features: int,
    net_type: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    X_test: np.ndarray,
    model_path: Optional[str],
    data_path_test: Optional[str],
    training_config: Dict[str, object] = getTrainingConfig(),
    model_config: Dict[str, object] = getModelConfig(),
) -> nn.Module:
    """Train a DistNet/BayesDistNet model.

    Inputs:
    - num_features: input feature dimension D
    - net_type: distnet or bayes_distnet
    - train_loader: training batches (inputs[B, D], observations[B, 2])
    - validation_loader: validation batches (inputs[B, D], observations[B, 2])
    - X_test: features [N_test, D] for periodic exportData path
    - model_path: unused path in memory-only checkpoint mode
    - data_path_test: unused path in memory-only export mode
    - training_config: optimization/loss settings
    - model_config: architecture/prior settings

    Outputs:
    - trained model with best-state restored and memory artifacts attached

    Semantic role:
    - Preserves legacy training control flow and scheduling while enforcing
      memory-only checkpointing and automatic best-state restore.
    """

    logger = logging.getLogger()
    logger.info("Using device %s", device.device)

    model_config["output_size"] = getNumberOfParameters(training_config["loss_fn"])
    model = getModel(net_type, model_config, num_features)

    logger.info(trainingConfigToStr(training_config))
    logger.info(modelConfigToStr(model_config))
    logger.info("Number of parameters: %s", count_parameters(model))

    model.apply(weights_init)
    model.to(device.device)

    start_rate = training_config["start_rate"]
    end_rate = training_config["end_rate"]
    n_epochs = training_config["n_epochs"]
    n_expected_epochs = training_config["n_expected_epochs"]
    n_optim = training_config["n_optim"]
    early_stop = training_config["early_stop"]

    if n_optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=start_rate, weight_decay=0.0001, momentum=0.95)
    elif n_optim == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=start_rate, weight_decay=0.001, amsgrad=False)
    else:
        raise ValueError("Unknown optimizer type.")

    decay_rate = np.exp(np.log(end_rate / start_rate) / n_expected_epochs)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
    logger.info("Decay rate: %s", decay_rate)

    total_len = len(train_loader) * training_config["batch_size"]
    counter = 1
    valid_loss_min = np.inf
    valid_loss_counter = 0

    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    # periodic_exports: Dict[int, np.ndarray] = {}
    # periodic_model_states: Dict[int, Dict[str, torch.Tensor]] = {}
    # history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        train_loss, train_kl = train_model(model, train_loader, optimizer, training_config, total_len, epoch, n_epochs)
        val_loss, log_loss, val_kl = validate_model(model, validation_loader, training_config, total_len, epoch, n_epochs)

        output_msg = (
            "Epoch: {:>4d}, Training Loss {:>12,.4f} Training KL {:>12,.4f}, "
            "Validation Loss {:>12,.4f}, Validation log {:>12,.4f}, Validation KL {:>12,.4f}"
        )
        print(output_msg.format(epoch + 1, train_loss, train_kl, val_loss, log_loss, val_kl))
        logger.info(output_msg.format(epoch + 1, train_loss, train_kl, val_loss, log_loss, val_kl))
        counter += 1

        # history.append(
        #     {
        #         "epoch": float(epoch + 1),
        #         "train_loss": float(train_loss),
        #         "train_kl": float(train_kl),
        #         "val_loss": float(val_loss),
        #         "val_log_loss": float(log_loss),
        #         "val_kl": float(val_kl),
        #     }
        # )

        # if epoch % 10 == 0:
        #     periodic_exports[epoch] = exportData(X_test, data_path_test, model)
        #     periodic_model_states[epoch] = saveModel(model, model_path)

        if val_loss < valid_loss_min:
            valid_loss_counter = 0
            valid_loss_min = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        else:
            valid_loss_counter += 1

        if valid_loss_counter >= early_stop:
            logger.info("Early stoppage, breaking")
            break

        lr_scheduler.step()

    model.load_state_dict(best_state_dict)
    model._memory_training_artifacts = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(valid_loss_min),
        "early_stop": int(early_stop),
        "stopped_by_patience": bool(valid_loss_counter >= early_stop),
        # "history": history,
        # "periodic_exports": periodic_exports,
        # "periodic_model_states": periodic_model_states,
    }
    return model


# ---------------------------------------------------------------------------
# Legacy post-hoc metrics from create_dataframes.py
# ---------------------------------------------------------------------------


N_STEPS = 200
DF_COLUMNS = [
    "Scenario",
    "Num Samples",
    "LB",
    "Seed",
    "Fold",
    "Mode",
    "Model",
    "LLH",
    "NLLH",
    "P-KS",
    "D-KS",
    "D-B",
    "KLD",
    "Var",
    "Mass",
]


def getDistribution(net_type: str, dist: str, params: np.ndarray):
    """Create scipy distribution object for a model output row.

    Inputs:
    - net_type: distnet or bayes_distnet
    - dist: one of EXPONENTIAL, INVGAUSS, LOGNORMAL, NORMAL, BAYESIAN_*
    - params: array-like distribution parameters or Bayesian samples

    Outputs:
    - scipy distribution object or gaussian_kde.

    Semantic role:
    - Exact legacy distribution factory for post-hoc metrics.
    """

    if net_type == "bayes_distnet":
        dist_obj = stats.gaussian_kde(params)
    elif dist == "EXPONENTIAL":
        dist_obj = stats.expon(loc=0, scale=params[0])
    elif dist == "INVGAUSS":
        dist_obj = stats.invgauss(params[0] / params[1], loc=0, scale=params[1])
    elif dist == "LOGNORMAL":
        dist_obj = stats.lognorm(s=params[0], loc=0, scale=params[1])
    elif dist == "NORMAL":
        dist_obj = stats.norm(loc=params[0], scale=params[1])
    else:
        raise ValueError("Unknown dist: {}".format(dist))
    return dist_obj


def getDistKS(dist: str, params: np.ndarray, runtimes: np.ndarray, net_type: str):
    """Run KS test for one instance.

    Inputs:
    - dist: distribution name string
    - params: model parameters or Bayesian samples
    - runtimes: empirical runtime samples [R]
    - net_type: distnet or bayes_distnet

    Outputs:
    - tuple (D_statistic, p_value)

    Semantic role:
    - Exact legacy KS helper.
    """

    if net_type == "bayes_distnet":
        return stats.ks_2samp(runtimes, params)
    if dist == "EXPONENTIAL":
        return stats.kstest(runtimes, "expon", [0, params[0]])
    if dist == "INVGAUSS":
        return stats.kstest(runtimes, "invgauss", [params[0] / params[1], 0, params[1]])
    if dist == "LOGNORMAL":
        return stats.kstest(runtimes, "lognorm", [params[0], 0, params[1]])
    if dist == "NORMAL":
        return stats.kstest(runtimes, "norm", [params[0], params[1]])
    raise ValueError("Unknown dist: {}".format(dist))


def getNLLH(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> float:
    """Compute mean negative log-likelihood over instances.

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: array [N_instances, R]
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - scalar mean NLLH

    Semantic role:
    - Preserved old NLLH post-hoc metric.
    """

    nllhs = []
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)

        temp = model_dist.pdf(instance)
        temp = np.array([np.log(i) if i > 1e-6 else np.log(1e-4) for i in temp], dtype=np.float64)
        nllh_per_instance = temp + np.log(max(instance))
        nllhs.append(-np.mean(nllh_per_instance))
    return float(np.mean(nllhs))


def getKSTest(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> Tuple[float, float]:
    """Compute aggregate KS rejection rate and KS distance.

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: array [N_instances, R]
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - (mean_rejection_indicator, mean_D_statistic)

    Semantic role:
    - Preserved old KS summary metric.
    """

    ps = []
    distances = []
    for params, instance in zip(model_params, fold_runtimes):
        d, p = getDistKS(dist, params, instance, net_type)
        distances.append(d)
        ps.append(1 if p < 0.01 else 0)
    return float(np.mean(ps)), float(np.mean(distances))


def getVariances(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> float:
    """Estimate mean predictive variance from sampled draws.

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: unused shape [N_instances, R], kept for signature
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - scalar mean variance

    Semantic role:
    - Preserved old variance diagnostic metric.
    """

    _ = fold_runtimes
    variances = []
    for params in model_params:
        model_dist = getDistribution(net_type, dist, params)
        samples = model_dist.resample(100) if net_type == "bayes_distnet" else model_dist.rvs(size=100)
        variances.append(np.var(samples, ddof=1))
    return float(np.mean(variances))


def getBhattacharyyaDistance(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> float:
    """Compute mean Bhattacharyya distance between empirical and model densities.

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: array [N_instances, R]
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - scalar mean distance

    Semantic role:
    - Preserved old D-B metric from create_dataframes.
    """

    distance = 0.0
    xs = np.linspace(0, 1.5, N_STEPS)
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        reference_kde = stats.gaussian_kde(instance)

        delta = 1.0 / N_STEPS
        ref_vals = np.asarray(reference_kde.pdf(xs), dtype=np.float64)
        model_vals = np.asarray(model_dist.pdf(xs), dtype=np.float64)
        BC = np.sum(np.sqrt(np.maximum(ref_vals * model_vals, 0.0))) * delta
        distance += -np.log(BC + 1e-8)
    return float(distance / len(model_params))


def getKLD(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> float:
    """Compute mean KL divergence from empirical KDE to model density.

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: array [N_instances, R]
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - scalar mean KL divergence

    Semantic role:
    - Preserved old KLD metric from create_dataframes.
    """

    kld = 0.0
    xs = np.linspace(0, 1.5, N_STEPS)
    epsilon = 1e-8
    delta = 1.0 / N_STEPS

    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        reference_kde = stats.gaussian_kde(instance)

        ref_vals = np.asarray(reference_kde.pdf(xs), dtype=np.float64)
        model_vals = np.asarray(model_dist.pdf(xs), dtype=np.float64)

        term = ref_vals * (np.log(ref_vals + epsilon) - np.log(model_vals + epsilon))
        kld += float(np.sum(term) * delta)

    return float(kld / len(model_params))


def getMass(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> float:
    """Compute mean probability mass outside [0, 1.5*max(instance)].

    Inputs:
    - model_params: array [N_instances, ...]
    - fold_runtimes: array [N_instances, R]
    - dist: distribution name
    - net_type: distnet or bayes_distnet

    Outputs:
    - scalar outside-mass mean

    Semantic role:
    - Preserved old mass diagnostic metric.
    """

    MIN_T = 0.0
    mass = 0.0
    for params, instance in zip(model_params, fold_runtimes):
        model_dist = getDistribution(net_type, dist, params)
        MAX_T = max(instance) * 1.5

        if net_type == "bayes_distnet":
            inside_mass = model_dist.integrate_box_1d(MIN_T, MAX_T)
        else:
            inside_mass = model_dist.cdf(MAX_T) - model_dist.cdf(MIN_T)

        mass += 1.0 - inside_mass
    return float(mass / len(model_params))


def calculate_posthoc_metrics(model_params: np.ndarray, fold_runtimes: np.ndarray, dist: str, net_type: str) -> Dict[str, float]:
    """Compute legacy post-hoc metric suite.

    Inputs:
    - model_params: model outputs [N_instances, ...]
    - fold_runtimes: empirical samples [N_instances, R]
    - dist: distribution name for non-Bayesian path
    - net_type: distnet or bayes_distnet

    Outputs:
    - dict with keys NLLH, P-KS, D-KS, D-B, KLD, Var, Mass.

    Semantic role:
    - Convenience wrapper around exact legacy metric functions.
    """

    ks_p, ks_distance = getKSTest(model_params, fold_runtimes, dist, net_type)
    return {
        "NLLH": getNLLH(model_params, fold_runtimes, dist, net_type),
        "P-KS": ks_p,
        "D-KS": ks_distance,
        "D-B": getBhattacharyyaDistance(model_params, fold_runtimes, dist, net_type),
        "KLD": getKLD(model_params, fold_runtimes, dist, net_type),
        "Var": getVariances(model_params, fold_runtimes, dist, net_type),
        "Mass": getMass(model_params, fold_runtimes, dist, net_type),
    }


# ---------------------------------------------------------------------------
# Adapter model for current project integration
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set numpy and torch RNG seeds.

    Inputs:
    - seed: integer random seed

    Outputs:
    - none

    Semantic role:
    - Ensures reproducible model initialization and sampling order.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _add_solution_flag(y: np.ndarray, lb: int = 0) -> np.ndarray:
    """Create legacy observation matrix with censoring flag.

    Inputs:
    - y: target array [N, 1] or [N]
    - lb: censoring lower-bound percent, 0 disables censoring

    Outputs:
    - observation array [N, 2] columns [target, sol_flag]

    Semantic role:
    - Inlines old censoring mechanism while keeping default off path.
    """

    y_vec = np.asarray(y, dtype=np.float32).reshape(-1)
    y_obs = np.c_[y_vec, np.ones(len(y_vec), dtype=np.float32)]
    if lb != 0:
        ytemp = copy.deepcopy(y_vec)
        ytemp.sort(axis=0)
        idx = int(len(ytemp) * (1 - (lb / 100)) - 1)
        censored_time = ytemp[idx]
        mask = y_vec > censored_time
        y_c = np.where(~mask, y_vec, censored_time)
        y_obs = np.c_[y_c, np.where(~mask, 1, 0).astype(np.float32).flatten()]
    return y_obs.astype(np.float32)


class BayesianDistNetModel:
    """Current-project adapter around legacy BayesDistNetFCN training path.

    Inputs:
    - n_input_features: feature width D
    - save_path: optional path retained for interface compatibility (memory-only)
    - X_valid, y_valid: optional validation arrays for metric tracking
    - lb: censoring percentage, default 0 disables censoring effect

    Outputs:
    - object exposing train(), predict(), and model state attributes

    Semantic role:
    - Isolates current-project API adaptation while preserving old internals,
      with fixed legacy hyperparameters (no external overrides).
    """

    def __init__(
        self,
        n_input_features: int,
        save_path: Optional[str] = None,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        lb: int = 0,
    ):
        self.model_target_scale = "max"
        self.n_input_features = n_input_features
        self.random_state = int(TRAINING_CONFIG_INI_DEFAULTS["seed"])
        self.save_path = save_path
        self.lb = int(lb)

        self.validation_available = X_valid is not None and y_valid is not None
        self.X_valid = np.asarray(X_valid, dtype=np.float32) if X_valid is not None else None
        self.y_valid = np.asarray(y_valid, dtype=np.float32) if y_valid is not None else None

        # Strict lock: always use the legacy Bayesian lognormal section defaults.
        self.fixed_training_config, self.fixed_model_config = parseConfig("BAYESIAN_LOGNORMAL")
        self.training_config = copy.deepcopy(self.fixed_training_config)
        self.model_config = copy.deepcopy(self.fixed_model_config)

        self.n_epochs = int(self.fixed_training_config["n_epochs"])
        self.batch_size = int(self.fixed_training_config["batch_size"])
        self.wc_time_limit = 0

        self.memory_exports: Dict[str, object] = {}
        self.best_epoch: Optional[int] = None
        self.model: Optional[nn.Module] = None
        self.last_sample_predictions: Optional[np.ndarray] = None

        set_seed(self.random_state)
        device.device = torch.device("cpu")

    @classmethod
    def load_model(cls, load_path: str, n_input_features: int) -> "BayesianDistNetModel":
        """Load model state from filesystem path into adapter object.

        Inputs:
        - load_path: path to a state_dict file
        - n_input_features: feature width D

        Outputs:
        - BayesianDistNetModel instance with initialized model

        Semantic role:
        - Compatibility utility for inference-only loading.
        """

        instance = cls(
            n_input_features=n_input_features,
        )
        instance.model_config["output_size"] = 1
        instance.model = BayesDistNetFCN(n_input_features, instance.model_config)
        state = torch.load(load_path, map_location="cpu")
        instance.model.load_state_dict(state)
        instance.model.eval()
        return instance

    def _build_loaders(self, X_train: np.ndarray, y_train_obs: np.ndarray) -> Tuple[DataLoader, DataLoader, np.ndarray]:
        """Build train and validation dataloaders for legacy train function.

        Inputs:
        - X_train: array [N, D]
        - y_train_obs: array [N, 2]

        Outputs:
        - train_loader: batches with drop_last=True
        - val_loader: validation batches
        - X_export_ref: feature array [N_ref, D] for periodic export

        Semantic role:
        - Adapter layer that prepares tensors without altering core train logic.
        """

        X_tr_t = torch.as_tensor(X_train, dtype=torch.float32, device=device.device)
        y_tr_t = torch.as_tensor(y_train_obs, dtype=torch.float32, device=device.device)

        train_dataset = TensorDataset(X_tr_t, y_tr_t)

        if len(train_dataset) == 0:
            raise ValueError("Empty training data after preprocessing/subsampling.")
        
        flag_batch_size = self.fixed_training_config["batch_size"]
        flag_drop_last = len(train_dataset) >= flag_batch_size

        train_loader = DataLoader(train_dataset, batch_size=flag_batch_size, shuffle=True, drop_last=flag_drop_last)

        if self.validation_available and self.X_valid is not None and self.y_valid is not None:
            y_val_obs = _add_solution_flag(self.y_valid, lb=0)
            X_val_t = torch.as_tensor(self.X_valid, dtype=torch.float32, device=device.device)
            y_val_t = torch.as_tensor(y_val_obs, dtype=torch.float32, device=device.device)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=flag_batch_size, shuffle=False)
            x_export = self.X_valid
        else:
            val_loader = DataLoader(train_dataset, batch_size=flag_batch_size, shuffle=False)
            x_export = X_train

        return train_loader, val_loader, x_export

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Bayesian DistNet model.

        Inputs:
        - X_train: array [N, D]
        - y_train: array [N, 1]

        Outputs:
        - none; model state updated in self.model

        Semantic role:
        - Uses preserved legacy train loop with in-memory checkpoint restore.
        """

        assert X_train.ndim == 2 and y_train.ndim == 2, "X_train and y_train must be 2D"
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have same number of rows"
        assert y_train.shape[1] == 1, "y_train must have shape [N, 1]"

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        y_train_obs = _add_solution_flag(y_train, lb=self.lb)
        train_loader, val_loader, x_export = self._build_loaders(X_train, y_train_obs)

        # Rebuild fixed configs on every call to block implicit overrides.
        fixed_training_cfg, fixed_model_cfg = parseConfig("BAYESIAN_LOGNORMAL")
        self.fixed_training_config = fixed_training_cfg
        self.fixed_model_config = fixed_model_cfg
        self.training_config = copy.deepcopy(self.fixed_training_config)
        self.model_config = copy.deepcopy(self.fixed_model_config)

        training_cfg = copy.deepcopy(self.fixed_training_config)
        model_cfg = copy.deepcopy(self.fixed_model_config)

        self.model = train(
            num_features=self.n_input_features,
            net_type="bayes_distnet",
            train_loader=train_loader,
            validation_loader=val_loader,
            X_test=x_export,
            model_path=self.save_path,
            data_path_test=self.save_path,
            training_config=training_cfg,
            model_config=model_cfg,
        )

        artifacts = getattr(self.model, "_memory_training_artifacts", {})
        self.memory_exports = artifacts
        self.best_epoch = artifacts.get("best_epoch")

    def predict_samples(self, X_test: np.ndarray, n_draws: Optional[int] = None, repeat_size: int = 64) -> np.ndarray:
        """Generate Bayesian sample predictions using legacy export protocol.

        Inputs:
        - X_test: array [N, D]
        - n_draws: number of stochastic forward draws (default training n_ens)
        - repeat_size: legacy batch replication factor per instance (default 64)

        Outputs:
        - sample array [N, repeat_size*n_draws]

        Semantic role:
        - Reproduces old exportData sampling behavior exactly.
        """

        if self.model is None:
            raise RuntimeError("Model is not trained.")

        draws = int(self.fixed_training_config["n_ens"]) if n_draws is None else int(n_draws)
        self.model.eval()

        X_test = np.asarray(X_test, dtype=np.float32)
        outputs: List[List[float]] = []
        for val_data in X_test:
            val_inputs = torch.tensor(val_data, dtype=torch.float32, device=device.device).unsqueeze(0)
            val_inputs = val_inputs.repeat(repeat_size, 1).to(device.device)
            rts: List[float] = []
            for _ in range(draws):
                val_pred = self.model(val_inputs)[0]
                rts.extend(val_pred.flatten().detach().cpu().tolist())
            outputs.append(rts)

        sample_array = np.array(outputs, dtype=np.float32)
        self.last_sample_predictions = sample_array
        return sample_array

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict distribution parameters for current-project metric API.

        Inputs:
        - X_test: array [N, D]

        Outputs:
        - params array [N, 2]
                    For fixed BAYESIAN_LOGNORMAL path: [sigma, mu_scale]

        Semantic role:
        - Interface-only adaptation from sample outputs to parametric format.
        """

        samples = self.predict_samples(X_test)
        outputs = torch.as_tensor(samples, dtype=torch.float32)

        safe_outputs = outputs + 1e-10
        mu = torch.exp(torch.log(safe_outputs).mean(dim=1, keepdim=True))
        sigma = torch.log(safe_outputs).std(dim=1, keepdim=True)
        params = torch.cat((sigma, mu), dim=1)

        return params.cpu().numpy()

    def evaluate_posthoc_metrics(self, y_test_scaled: np.ndarray) -> Dict[str, float]:
        """Evaluate legacy post-hoc metrics on latest sample predictions.

        Inputs:
        - y_test_scaled: array [N_instances, R] in legacy target scaling space

        Outputs:
        - metric dict with NLLH, KS, KLD, D-B, Var, Mass

        Semantic role:
        - Exposes old create_dataframes metrics in adapter API.
        """

        if self.last_sample_predictions is None:
            raise RuntimeError("No sample predictions available. Call predict_samples first.")

        dist_name = "BAYESIAN_LOGNORMAL"
        return calculate_posthoc_metrics(self.last_sample_predictions, y_test_scaled, dist_name, "bayes_distnet")


# Convenience alias for concise imports.
BayesDistNetModel = BayesianDistNetModel
