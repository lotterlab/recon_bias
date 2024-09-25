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
        early_stopping_patience=None,
    ):
        """
        Trainer class for training and validating a model with early stopping.

        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            num_epochs (int): Maximum number of epochs to train.
            device (torch.device): Device to run the training on (CPU or GPU).
            log_dir (str): Directory to save TensorBoard logs.
            output_dir (str): Directory to save model checkpoints.
            output_name (str, optional): Base name for the saved model files. Defaults to "model".
            save_interval (int, optional): Interval (in epochs) to save model checkpoints. Defaults to 1.
            early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped. If None, early stopping is disabled.
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
        self.early_stopping_patience = early_stopping_patience

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Create output directory for checkpoints
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.snapshot_dir = os.path.join(self.output_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Initialize early stopping variables
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None  # To store the best model's state_dict
        self.best_epoch = None  # To store the epoch number of the best model

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Training phase
            train_loss, train_metric = self.train_epoch(epoch)

            # Validation phase
            val_loss, val_metric = self.validate_epoch(epoch)

            # Log metrics
            self.writer.add_scalars(
                "Loss", {"Train": train_loss, "Validation": val_loss}, epoch
            )
            self.writer.add_scalars(
                f"{self.model.performance_metric_name}", {"Train": train_metric, "Validation": val_metric}, epoch
            )

            # Save checkpoint at specified intervals
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)
                self.save_snapshot(epoch)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()  # Store the model's state_dict
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if (
                    self.early_stopping_patience is not None
                    and self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    print(
                        f"Early stopping triggered after {self.early_stopping_patience} epochs with no improvement."
                    )
                    break

        # Save the final model
        self.save_checkpoint(epoch, final=True)

        # Save the best model
        if self.best_model_state is not None:
            self.save_best_model()

        self.writer.close()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_metrics = 0
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
            running_metrics += self.model.performance_metric(outputs, labels)
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(
                {
                    "Loss": running_loss / total,
                    f"{self.model.performance_metric_name}": running_metrics.double().item() / total,
                }
            )

        epoch_loss = running_loss / total
        epoch_metric = running_metrics.double() / total

        return epoch_loss, epoch_metric

    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_metrics = 0
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
                running_metrics += self.model.performance_metric(outputs, labels)
                total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix(
                    {
                        "Val Loss": running_loss / total,
                        f"Val {self.model.performance_metric_name}": running_metrics.double().item() / total,
                    }
                )

        epoch_loss = running_loss / total
        epoch_metric = running_metrics.double() / total

        return epoch_loss, epoch_metric
    
    def save_snapshot(self, epoch):
        """
        Saves a snapshot of the model at the current epoch.

        Args:
            epoch (int): Current epoch number.
        """
        train_snapshot_name = f"{self.output_name}_epoch_{epoch}_snapshot_train"
        train_iter = iter(self.train_loader)

        # Save snapshot for training data
        x, y = next(train_iter) 
        if len(x.shape) > 2 and x.shape[0] > 1: 
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad(): 
            y_pred = self.model(x)
            path = os.path.join(self.snapshot_dir, train_snapshot_name)
            self.model.save_snapshot(x, y, y_pred, path, self.device, epoch)

        
        # Save snapshot for validation data
        val_snapshot_name = f"{self.output_name}_epoch_{epoch}_snapshot_val"
        val_iter = iter(self.val_loader)

        x, y = next(val_iter) 
        if len(x.shape) > 2 and x.shape[0] > 1: 
            x = x[0].unsqueeze(0)
            y = y[0].unsqueeze(0)
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad(): 
            y_pred = self.model(x)
            path = os.path.join(self.snapshot_dir, val_snapshot_name)
            self.model.save_snapshot(x, y, y_pred, path, self.device, epoch)


    def save_checkpoint(self, epoch, final=False):
        """
        Saves the model checkpoint.

        Args:
            epoch (int): Current epoch number.
            final (bool, optional): If True, saves the model as the final model. Defaults to False.
        """
        if final:
            checkpoint_filename = f"{self.output_name}_epoch_{epoch}_final.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(self.model.state_dict(), checkpoint_path)
        else:
            checkpoint_filename = f"{self.output_name}_epoch_{epoch}.pth"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(self.model.state_dict(), checkpoint_path)

    def save_best_model(self):
        """
        Saves the best model (based on validation loss) to disk.
        """
        best_model_filename = f"{self.output_name}_epoch_{self.best_epoch}_best.pth"
        best_model_path = os.path.join(self.checkpoint_dir, best_model_filename)
        torch.save(self.best_model_state, best_model_path)
