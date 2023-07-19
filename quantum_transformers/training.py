import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch


def train(model, train_dataloader, valid_dataloader, learning_rate, num_epochs, device, verbose=False) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_auc, best_epoch = 0.0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        step = 0

        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.long().to(device)

            if verbose:
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
        y_true, y_score = [], []
        with torch.no_grad():
            val_loss = 0.0
            val_steps = 0
            val_correct = 0
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_steps += 1
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                y_true.append(labels.cpu().numpy())
                y_score.append(outputs.softmax(dim=1).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        if y_score.shape[1] == 2:  # binary classification
            y_score = y_score[:, 1]

        val_loss /= val_steps
        val_acc = val_correct / len(valid_dataloader.dataset)*100
        val_auc = roc_auc_score(y_true, y_score, multi_class='ovr') * 100
        print(f"Epoch {epoch+1}/{num_epochs} ({time.time()-start_time:.2f}s):",
              f"Loss = {val_loss:.4f},",
              f"Accuracy = {val_acc:.2f}%,",
              f"AUC = {val_auc:.2f}%")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1

    print(f"BEST AUC = {best_val_auc:.2f}% AT EPOCH {best_epoch}")

