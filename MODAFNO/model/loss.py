import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)

class RMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

class LogRMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        log_pred = torch.log(y_pred + self.eps)
        log_true = torch.log(y_true + self.eps)
        return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        log_pred = torch.log(y_pred + self.eps)
        log_true = torch.log(y_true + self.eps)
        return torch.mean((log_pred - log_true) ** 2)
    
class CharbonnierLoss(nn.Module):
    """
    Smooth L1 approximation: L = mean( sqrt((y_pred - y_true)^2 + eps^2) )
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        # add small eps^2 for numerical stability
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class GeneralPM25Loss(nn.Module):
    def __init__(self, loss_type='rmse', log_weight=0.5, eps=1e-6):
        """
        Parameters:
        - loss_type: 'mse', 'rmse', 'logrmse', 'logmse', 'charbonnier' (or 'charb'), or 'combined'
        - log_weight: weight of log-based loss in combined mode
        - eps: small value to avoid log(0) and stabilize Charbonnier
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps
        self.log_weight = log_weight

        valid = {'mse', 'rmse', 'logrmse', 'logmse', 'combined', 'charbonnier', 'charb'}
        if self.loss_type not in valid:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        if self.loss_type == 'mse':
            self.mse = MSELoss()
            self.forward = self._mse_forward

        elif self.loss_type == 'rmse':
            self.rmse = RMSELoss()
            self.forward = self._rmse_forward

        elif self.loss_type == 'logrmse':
            self.logrmse = LogRMSELoss(eps=eps)
            self.forward = self._logrmse_forward

        elif self.loss_type == 'logmse':
            self.logmse = LogMSELoss(eps=eps)
            self.forward = self._logmse_forward

        elif self.loss_type in {'charbonnier', 'charb'}:
            self.charb = CharbonnierLoss(eps=eps)
            self.forward = self._charbonnier_forward

        elif self.loss_type == 'combined':
            self.rmse = RMSELoss()
            self.logrmse = LogRMSELoss(eps=eps)
            self.forward = self._combined_forward

    def _mse_forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)

    def _rmse_forward(self, y_pred, y_true):
        return self.rmse(y_pred, y_true)

    def _logrmse_forward(self, y_pred, y_true):
        return self.logrmse(y_pred, y_true)

    def _logmse_forward(self, y_pred, y_true):
        return self.logmse(y_pred, y_true)

    def _charbonnier_forward(self, y_pred, y_true):
        return self.charb(y_pred, y_true)

    def _combined_forward(self, y_pred, y_true):
        return (
            (1 - self.log_weight) * self.rmse(y_pred, y_true)
            + self.log_weight * self.logrmse(y_pred, y_true)
        )
