from collections import defaultdict
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
import random
from re import T
import segmentation_models_pytorch as pytorch_models
from skimage.color import label2rgb
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from typing import Optional
from src.data.segmentation_dataset import SegmentationDataset
from src.utils.transformations import min_max_slice_normalization


def l1_loss(ground_truth, prediction):
    # TASK: compute the L1 loss ---------------------------------------------------------------
    # loss
    # -----------------------------------------------------------------------------------------
    absolute_differences = torch.abs(prediction - ground_truth)

    loss = torch.mean(absolute_differences)

    return loss


def dice_score(ground_truth, prediction):
    # TASK: compute the dice score ---------------------------------------------------------------
    # score
    # --------------------------------------------------------------------------------------------

    ground_truth = ground_truth.view(-1)
    prediction = prediction.view(-1)

    intersection = (prediction * ground_truth).sum()
    total = prediction.sum() + ground_truth.sum()

    return (2.0 * intersection) / (total)


def dice_loss(ground_truth, prediction):
    # TASK: compute the dice loss ---------------------------------------------------------------
    # loss
    # -------------------------------------------------------------------------------------------
    return 1 - dice_score(ground_truth, prediction)


class TumorSegmentation(pl.LightningModule):
    def __init__(self, model, lr, loss):
        super().__init__()
        self.backbone = model
        self.lr = lr
        self.loss = loss
        self.metric = dice_score
        self.writer = SummaryWriter()

    def forward(self, x):
        y = self.backbone(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.unsqueeze(dim=1)
        # TASK: perform the forward pass, compute the loss and the metric of each step --------------
        # y_pred: the prediction of the network
        # loss
        # metric
        # -------------------------------------------------------------------------------------------
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        metric = self.metric(y, y_pred)

        return {"loss": loss, "metric": metric}

    def on_train_epoch_end(self, output):
        loss = 0
        metric = 0
        for o in output:
            # TASK: compute the loss and metric of the epoch -------------------------------------------
            # loss
            # metric
            # ------------------------------------------------------------------------------------------
            loss += o["loss"]
            metric += o["metric"]
        loss = loss / len(output)
        metric = metric / len(output)
        self.writer.add_scalar("Epoch_loss/training", loss, self.current_epoch)
        self.writer.add_scalar("Epoch_metric/training", metric, self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.unsqueeze(dim=1)
        # TASK: perform the forward pass, compute the loss and the metric of each step --------------
        # y_pred: the prediction of the network
        # loss
        # metric
        # -------------------------------------------------------------------------------------------
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        metric = self.metric(y, y_pred)

        y_pred_plot = np.array(y_pred.cpu(), dtype=float)
        image = self.prepare_visualization(
            y[0, 0, :, :].cpu().numpy(),
            y_pred_plot[0, 0, :, :],
            x[0, 0, :, :].cpu().numpy(),
        )
        fig, ax = plt.subplots(nrows=1, ncols=4)
        ax[0].imshow(x[0, 0, :, :].cpu().numpy(), cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(y[0, 0, :, :].cpu().numpy().astype("uint8"), cmap="gray")
        ax[1].set_title("Ground Truth Seg")
        ax[2].imshow(y_pred_plot[0, 0, :, :].astype("uint8"), cmap="gray")
        ax[2].set_title("Predicted Segm")
        ax[3].imshow(y_pred_plot[0, 0, :, :].astype("uint8"), cmap="gray", alpha=0.5)
        ax[3].imshow(
            y[0, 0, :, :].cpu().numpy().astype("uint8"), cmap="gray", alpha=0.5
        )
        ax[3].set_title("Overlay")
        self.writer.add_figure("Validation/" + str(batch_idx), fig, self.current_epoch)
        plt.close()
        return {"loss": loss, "metric": metric}

    def prepare_visualization(self, y, y_pred, image):
        annotation_pred = (y_pred > 0.5).astype(
            "uint8"
        )  # It will evaluate the logical expression y_predict>0.25 and return True or False
        annotation_pred = np.uint8(annotation_pred)
        annotation_gt = np.uint8(y)

        overlay = np.copy(image)
        image_label_overlay = label2rgb(
            annotation_pred, image=overlay, bg_label=0, alpha=0.5, colors=["red"]
        )

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        redImg = np.zeros(image.shape, image.dtype)
        redImg[:, :] = (0, 255, 0)
        redMask = cv2.bitwise_and(redImg, redImg, mask=annotation_gt)
        image_mask1 = np.float32(image_label_overlay)
        image = cv2.addWeighted(redMask, 0.05, image_mask1, 0.95, 0.0)
        return image

    def on_validation_epoch_end(self, output):
        loss = 0
        metric = 0
        for o in output:
            # TASK: compute the loss and metric of the epoch -------------------------------------------
            # metric
            # loss
            # ------------------------------------------------------------------------------------------
            loss += o["loss"]
            metric += o["metric"]
        loss = loss / len(output)
        metric = metric / len(output)
        self.log("val_dice", metric)
        self.writer.add_scalar("Epoch_loss/validation", loss, self.current_epoch)
        self.writer.add_scalar("Epoch_metric/validation", metric, self.current_epoch)


unet5 = pytorch_models.Unet(
    in_channels=1,
    classes=1,
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    encoder_weights=None,
    activation="sigmoid",
)

transform = transforms.Compose([min_max_slice_normalization])

train_dataset = SegmentationDataset(
    data_root="../../data/UCSF-PDGM",
    transform=transform,
    split="train",
    number_of_samples=8,
    seed=42,
    type="T2",
)
val_dataset = SegmentationDataset(
    data_root="../../data/UCSF-PDGM",
    transform=transform,
    split="val",
    number_of_samples=8,
    seed=42,
    type="T2",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=1,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
)


net = TumorSegmentation(unet5, lr=0.002, loss=l1_loss)

trainer = pl.Trainer(
    precision=16,
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
    max_epochs=10,  # TASK: modify this value and see how the results are changing
)

# Train!
trainer.fit(net, train_loader, val_loader)
