import torch
import torch.nn as nn

class NLLSurvLoss(nn.Module):
    """
    Calculates the negative log likelihood loss for survival prediction with
    censoring.

    Attributes
    ----------
    eps :  float, optional
        A small value to prevent numerical instability. Defaults to 1e-8.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, label):
        """
         Computes the loss for a batch of data.

        Parameters
        ----------
        logits : torch.Tensor, shape (batch_size, num_time_steps)
            Predicted logits from the model
        label :  torch.Tensor, shape (batch_size,)
            Observed event times (or censoring times)
        censor : torch.Tensor, shape (batch_size,)
            Censoring indicators--1 for censored events (i.e., alive) and
            0 for uncensored (i.e., deceased).

         Returns
         -------
             torch.Tensor, shape (1,)
             The mean negative log likelihood loss.
        """
        censor = label[:,4]
        censor = censor.unsqueeze(1)
        os = (label[:, 5] // 250).long()
        os = os.unsqueeze(1)
        # Compute hazard, survival, and offset survival
        haz = logits.sigmoid() + self.eps  # prevent log(0) downstream
        sur = torch.cumprod(1 - haz, dim=1)
        sur_pad = torch.cat([torch.ones_like(censor), sur], dim=1)

        # Get values at ground truth bin
        sur_pre = sur_pad.gather(dim=1, index=os)
        sur_cur = sur_pad.gather(dim=1, index=os + 1)
        haz_cur = haz.gather(dim=1, index=os)

        # Compute NLL loss
        loss = (
            -(1 - censor) * sur_pre.log()
            - (1 - censor) * haz_cur.log()
            - censor * sur_cur.log()
        )

        return loss.mean()