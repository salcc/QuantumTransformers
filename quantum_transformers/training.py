import time

import torch
import torch.utils.data
import torchmetrics.classification


def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, valid_dataloader: torch.utils.data.DataLoader, num_classes: int,
          learning_rate: float, num_epochs: int, device: torch.device, verbose: bool = False) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_auc, best_epoch = 0.0, 0
    if num_classes == 2:
        auc_metric = torchmetrics.classification.BinaryAUROC()
    else:
        auc_metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_classes)
    start_time = time.time()
    for epoch in range(num_epochs):
        step = 0

        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.long().to(device)


            operation_start_time = time.time()

            optimizer.zero_grad()

            if verbose:
                print(f" Zero grad ({time.time()-operation_start_time:.2f}s)")
                operation_start_time = time.time()

            outputs = model(inputs)

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
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} ({time.time()-start_time:.2f}s): Step {step}, Loss = {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                if num_classes == 2:
                    probabilities = torch.sigmoid(outputs[:, 1])
                else:
                    probabilities = torch.softmax(outputs, dim=1)
                auc_metric.update(probabilities, labels)

        val_loss /= len(valid_dataloader)
        val_auc = auc_metric.compute() * 100

        print(f"Epoch {epoch+1}/{num_epochs} ({time.time()-start_time:.2f}s):",
              f"Loss = {val_loss:.4f},",
              f"AUC = {val_auc:.2f}%")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1

    print(f"BEST AUC = {best_val_auc:.2f}% AT EPOCH {best_epoch}")
