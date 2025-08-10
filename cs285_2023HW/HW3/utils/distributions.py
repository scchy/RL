# python3
# Create Date: 2025-08-10
# Func: 
#   cs285.infrastructure.distributions
# reference: 
#    https://github.com/pytorch/rl/blob/main/torchrl/modules/distributions/continuous.py
# =============================================================================

import torch
import torch.distributions as D
import math
from numbers import Number
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all
from typing import Union


def make_multi_normal(
        mean: torch.Tensor,
        std: Union[float, torch.Tensor]
) -> D.Distribution:
    if isinstance(std, float):
        std = torch.tensor(std, device=mean.device)

    if std.shape == ():
        std = std.expand(mean.shape)

    return D.Independent(
        D.Normal(mean, std), 
        reinterpreted_batch_ndims=1
    )


def make_tanh_transformed(
        mean: torch.Tensor,
        std: Union[float, torch.Tensor]
) -> D.Distribution:
    if isinstance(std, float):
        std = torch.tensor(std, device=mean.device)

    if std.shape == ():
        std = std.expand(mean.shape)

    return D.Independent(
        D.TransformedDistribution(
            base_distribution=D.Normal(mean, std),
            transforms=[D.Transform(cache_size=1)]
        ),
        reinterpreted_batch_ndims=1
    )


def make_truncated_normal(
    mean: torch.Tensor, std: Union[float, torch.Tensor]
) -> D.Distribution:
    if isinstance(std, float):
        std = torch.tensor(std, device=mean.device)

    if std.shape == ():
        std = std.expand(mean.shape)

    return D.Independent(
        TruncatedNormal(
            mean,
            std,
            -1.0,
            1.0,
        ),
        reinterpreted_batch_ndims=1,
    )


CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


# https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw3/cs285/infrastructure/distributions.py
class TruncatedStandardNormal(Distribution):
    """Truncated Standard Normal distribution.

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """


class TruncatedNormal(TruncatedStandardNormal):
    """Truncated Normal distribution.

    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
