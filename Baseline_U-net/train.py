from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import torch
import torch.nn as nn
from datetime import datetime


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- Entrenamiento bï¿½sico ---

def train(model, train_loader, val_loader, epochs, device):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_model_name = f"outputs/model_{timestamp}.pth"

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-8)
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        val_loss = 0

        total_steps = len(train_loader) + len(val_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]Epoch {epoch}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Entrenando y Validando", total=total_steps)

            # Entrenamiento
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

            # Validaciï¿½n
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).float()
                    preds = model(batch_X)
                    loss = loss_fn(preds, batch_y) 
                    val_loss += loss.item() * batch_X.size(0)
                    progress.update(task, advance=1)

            val_loss /= len(val_loader.dataset)

        print(f"\nEpoch {epoch}: Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_name)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping en epoch {epoch}")
            break