import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from scipy.optimize import basinhopping


# Class weights computed using train set
# log2(number of data of all classes / number of data in one class)
CLASSES_WEIGHTS = np.log2([
    6678.667, 17.826, 1252.25, 5.759, 17.826, 166.967,
    313.062, 385.308, 1252.25, 70.056, 141.099, 801.44,
    36.363, 801.44, 572.457, 626.125, 871.13, 48.396,
    3.806, 910.727, 2.109, 2862.286, 67.01, 1252.25,
    690.897, 589.294, 36.899, 63.809, 22.238, 47.933,
    4.093, 159.016, 572.457, 333.933
])


class FocalLoss(nn.Module):

    def __init__(self, gamma=1, weight=None, balance=0.75):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.balance = balance
        return

    def forward(self, inputs, target):
        logpt = -F.binary_cross_entropy_with_logits(
            input=inputs, target=target, reduction='none')
        if self.weight is not None:
            logpt = logpt * self.weight
        logpt = logpt.mean()
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance * focal_loss
        return balanced_focal_loss


class FBetaLoss(nn.Module):

    def __init__(self, beta=1):
        super(FBetaLoss, self).__init__()

        self.eps = 1e-8
        self.beta = beta
        self.beta2 = beta ** 2
        return

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        tp = (inputs * target).sum(dim=1)
        precision = tp.div(inputs.sum(dim=1).add(self.eps))
        recall = tp.div(target.sum(dim=1).add(self.eps))

        fbeta = torch.mean(
            (precision * recall)
            .div(precision.mul(self.beta2) + recall + self.eps)
            .mul(1 + self.beta2)
        )
        return 1 - fbeta


class ComboLoss(nn.Module):

    def __init__(self, losses, weights):
        super(ComboLoss, self).__init__()

        assert len(losses) == len(weights),\
            'losses and weights should have same length'

        classes_weights = torch.tensor(CLASSES_WEIGHTS).float()
        if torch.cuda.is_available():
            classes_weights = classes_weights.cuda()

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=classes_weights)
        mlsml_loss = nn.MultiLabelSoftMarginLoss(weight=classes_weights)
        focal_loss = FocalLoss(gamma=1, weight=classes_weights, balance=0.75)
        f1_loss = FBetaLoss(beta=1)

        self.losses_weights = []
        for loss, weight in zip(losses, weights):
            if loss == 'mlsml':
                item = [mlsml_loss, weight]
            elif loss == 'bce':
                item = [bce_loss, weight]
            elif loss == 'focal':
                item = [focal_loss, weight]
            elif loss == 'f1':
                item = [f1_loss, weight]
            else:
                continue
            self.losses_weights.append(item)
        return

    def forward(self, inputs, targets):
        total_loss = 0
        for loss, weight in self.losses_weights:
            total_loss += weight * loss(inputs, targets)
        return total_loss


def best_f1_score(labels, preds):

    def f1_neg(threshold):
        return -f1_score(labels.flatten(),
                         (preds > threshold).flatten())

    def bounds(**kwargs):
        x = kwargs['x_new']
        tmax = bool(np.all(x <= 1))
        tmin = bool(np.all(x >= 0))
        return tmax and tmin

    def print_func(x, f, accepted):
        print('at minimum {:.6f} accepted {}'.format(f, int(accepted)))

    print('[Finding best threshold ...]')
    n = labels.shape[1]
    thr_0 = [0.10] * n
    constraints = [(0., 1.)] * n
    minimizer_kwargs = {'method': 'L-BFGS-B',
                        'bounds': constraints,
                        'options': {'eps': 0.05}}
    opt_output = basinhopping(f1_neg, thr_0, stepsize=0.1,
                              minimizer_kwargs=minimizer_kwargs,
                              niter=10, accept_test=bounds,
                              callback=print_func)
    score = -opt_output.fun
    return score, opt_output.x
