import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        device,
        log_dir,
        output_dir,
        output_name="model",
        save_interval=1,
    ):
        """
        Trainer class for training and validating a model.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            num_epochs (int): Number of epochs to train.
            device (torch.device): Device to run the training on (CPU or GPU).
            log_dir (str): Directory to save TensorBoard logs.
            output_dir (str): Directory to save model checkpoints.
            output_name (str, optional): Base name for the saved model files. Defaults to "model".
            save_interval (int, optional): Interval (in epochs) to save model checkpoints. Defaults to 1.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.output_name = output_name
        self.save_interval = save_interval

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Create output directory for checkpoints
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)

            # Log metrics
            self.writer.add_scalars(
                "Loss", {"Train": train_loss, "Validation": val_loss}, epoch
            )
            self.writer.add_scalars(
                "Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch
            )

            # Save checkpoint at specified intervals
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)

        # Save the final model
        self.save_checkpoint(self.num_epochs, final=True)
        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        train_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Training]"
        )
        for inputs, labels in train_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.model.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = self.model.classification_criteria(outputs)
            transformed_labels = self.model.target_transformation(labels)
            running_corrects += torch.sum(preds == transformed_labels)
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(
                {
                    "Loss": running_loss / total,
                    "Acc": running_corrects.double().item() / total,
                }
            )

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        val_bar = tqdm(
            self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Validation]"
        )
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = self.model.classification_criteria(outputs)
                transformed_labels = self.model.target_transformation(labels)
                running_corrects += torch.sum(preds == transformed_labels)
                total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix(
                    {
                        "Val Loss": running_loss / total,
                        "Val Acc": running_corrects.double().item() / total,
                    }
                )

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, final=False):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): Current epoch number.
            final (bool, optional): If True, saves the model as the final model. Defaults to False.
        """
        if final:
            checkpoint_filename = f"{self.output_name}_final.pth"
            checkpoint_path = os.path.join(self.output_dir, checkpoint_filename)
        else:
            checkpoint_filename = f"{self.output_name}_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        torch.save(self.model.state_dict(), checkpoint_path)
