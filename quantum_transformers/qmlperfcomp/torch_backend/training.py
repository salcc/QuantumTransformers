import time

import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_and_evaluate(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, num_classes: int,
                       num_epochs: int, device: torch.device, learning_rate: float = 1e-3, verbose: bool = False) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_val_auc, best_epoch = 0.0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        step = 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format= '{l_bar}{bar:10}{r_bar}{bar:-10b}') as progress_bar:
            model.train()
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                if num_classes == 2:
                    labels = labels.float()
                else:
                    labels = labels.long()

                operation_start_time = time.time()

                optimizer.zero_grad()

                if verbose:
                    print(f" Zero grad ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                outputs = model(inputs)

                if num_classes == 2 and outputs.shape[1] == 2:
                    outputs = outputs[:, 1]

                if verbose:
                    print(f" Forward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                loss = criterion(outputs, labels)

                if verbose:
                    print(f" Loss ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                loss.backward()

                if verbose:
                    print(f" Backward ({time.time()-operation_start_time:.2f}s)")
                    operation_start_time = time.time()

                optimizer.step()

                if verbose:
                    print(f" Optimizer step ({time.time()-operation_start_time:.2f}s)")

                step += 1
                progress_bar.update(1)
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} ({time.time()-start_time:.2f}s): Step {step}, Loss = {loss.item():.4f}")

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if num_classes == 2:
                        labels = labels.float()
                    else:
                        labels = labels.long()
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    if num_classes == 2:
                        if outputs.shape[1] == 2:
                            outputs = outputs[:, 1]
                        probabilities = torch.sigmoid(outputs)
                    else:
                        probabilities = torch.softmax(outputs, dim=1)
                    y_true.extend(labels.tolist())
                    y_pred.extend(probabilities.tolist())

            val_loss /= len(valid_dataloader)
            val_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')

            progress_bar.set_postfix_str(f"Loss = {val_loss:.4f}, AUC = {val_auc:.2f}%")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1

    print(f"TOTAL TIME = {time.time()-start_time:.2f}s")
    print(f"BEST AUC = {best_val_auc:.2f}% AT EPOCH {best_epoch}")
