import torch
import torch.nn as nn
import torch.nn.functional as F

bce_loss = nn.BCEWithLogitsLoss(reduction="sum")


def criterion(prediction, mask, regr, regression_weight=0.99):

    # Binary mask loss
    pred_mask = prediction[:, 0]
    mask_loss = bce_loss(pred_mask, mask) / prediction.shape[0]

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(
        1
    ).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = (1 - regression_weight) * mask_loss + regression_weight * regr_loss

    return loss, mask_loss, regr_loss
