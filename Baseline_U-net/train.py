from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F

# --- Gradient-based loss function ---
def gradient_loss(y_true, y_pred):
    """
    Computes L1 loss between spatial gradients of target and prediction.
    Encourages sharper predictions with correct edge structure.
    """
    # Spatial gradients (x and y directions)
    dx_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]

    dx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

    # Mean absolute difference
    loss_x = torch.mean(torch.abs(dx_true - dx_pred))
    loss_y = torch.mean(torch.abs(dy_true - dy_pred))
    return loss_x + loss_y


class CombinedLoss(torch.nn.Module):
    """
    Combines MSE loss with gradient loss.
    Helps preserve fine spatial structures in super-resolution tasks.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        grad_loss = gradient_loss(y_true, y_pred)
        return mse_loss + self.alpha * grad_loss


class EarlyStopping:
    """
    Stops training early if validation loss does not improve.
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # First evaluation → set best_loss
        if self.best_loss is None:
            self.best_loss = val_loss
        # No improvement (within tolerance)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # Improvement → reset counter
        else:
            self.best_loss = val_loss
            self.counter = 0


# --- Training Loop ---
def train(model, train_loader, val_loader, epochs, device):
    """
    Full training pipeline with progress bar, validation, scheduler, early stopping, and checkpoint saving.
    """
    # Model checkpoint filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_model_name = f"outputs/model_{timestamp}.pth"

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8
    )
    loss_fn = CombinedLoss(alpha=1.0)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        val_loss = 0
        total_steps = len(train_loader) + len(val_loader)

        # Progress bar for training + validation
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]Epoch {epoch}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Training and Validation", total=total_steps)

            # --- Training loop ---
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                progress.update(task, advance=1)

            train_loss /= len(train_loader.dataset)

            # --- Validation loop ---
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
                    preds = model(batch_X)
                    loss = loss_fn(preds, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    progress.update(task, advance=1)

            val_loss /= len(val_loader.dataset)

        # Log results
        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_name)

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
