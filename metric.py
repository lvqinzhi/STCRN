"""Evaluation metrics used by STCRN."""

import torch


def _apply_mask(prediction, target, mask_value):
    if mask_value is None:
        return prediction, target
    mask = target > mask_value
    return torch.masked_select(prediction, mask), torch.masked_select(target, mask)


def mae(prediction, target, mask_value=None):
    prediction, target = _apply_mask(prediction, target, mask_value)
    return torch.mean(torch.abs(target - prediction))


def rmse(prediction, target, mask_value=None):
    prediction, target = _apply_mask(prediction, target, mask_value)
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def mape(prediction, target, mask_value=None):
    prediction, target = _apply_mask(prediction, target, mask_value)
    return torch.mean(torch.abs((target - prediction) / target))
