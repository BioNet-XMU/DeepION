from kornia.augmentation import *

from typing import Any, Dict, Optional
import numpy as np
import torch
from torch import Tensor

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


class IntensityDependentMissing(IntensityAugmentationBase2D):

    def __init__(
            self,
            same_on_batch: bool = False,
            p: float = 0.5,
            keepdim: bool = False,
            return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )

    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def apply_transform(
            self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:

        s, _, m, n = input.shape

        # print(input.shape)

        qqq = np.random.randint(10, 90)

        for u in range(s):
            bb = torch.quantile(input[u, :, :, :], qqq / 100)
            bb = torch.where(input[u, :, :, :] >= bb, input[u, :, :, :], torch.zeros_like(input[u, :, :, :]))
            input[u, :, :, :] = bb

        return input

class RandomMissing(IntensityAugmentationBase2D):

    def __init__(
            self,
            same_on_batch: bool = False,
            p: float = 0.5,
            keepdim: bool = False,
            return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(
            p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim
        )

    def generate_parameters(self, shape: torch.Size) -> Dict[str, Tensor]:
        noise = torch.randn(shape)
        return dict(noise=noise)

    def apply_transform(
            self, input: Tensor, params: Dict[str, Tensor], flags: Dict[str, Any], transform: Optional[Tensor] = None
    ) -> Tensor:

        s, _, m, n = input.shape

        qqq = np.random.randint(0, 800)

        ppp = m * n * qqq // 1000

        for i in range(ppp):
            ii = np.random.randint(0, m - 1)
            jj = np.random.randint(0, n - 1)
            input[:, :, ii, jj] = 0

        return input
