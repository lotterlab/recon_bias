import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, log_dir):
    """
    Trains a classification model and logs progress to TensorBoard.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss function): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on (CPU or GPU).
        log_dir (str): Directory to save TensorBoard logs.

    Returns:
        torch.nn.Module: The trained model.
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            #labels = {key: value.to(device) for key, value in labels.items()}
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            print(preds)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == (labels[:, 5] // 250).long())
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix({
                'Loss': running_loss / total,
                'Acc': running_corrects.double().item() / total
            })

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        # Log training metrics
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == (labels[:, 5] // 250).long())
                val_total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix({
                    'Val Loss': val_running_loss / val_total,
                    'Val Acc': val_running_corrects.double().item() / val_total
                })

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_running_corrects.double() / val_total

        # Log validation metrics
        writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_epoch_acc, epoch)

    writer.close()
    return model
