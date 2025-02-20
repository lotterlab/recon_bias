import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
import numpy as np
import torch.nn.functional as F

class FairnessLoss(nn.Module):
    def __init__(self, classifier, fairness_lambda=1, momentum=0.1, temperature=0.3):
        """
        Initialize the fairness loss module.
        
        Args:
            classifier: Pre-trained classifier model.
            fairness_lambda: Weight for the fairness loss.
            momentum: Momentum for updating running threshold (default: 0.1).
            temperature: Temperature for the smooth approximation of the threshold (default: 0.1).
        """
        super(FairnessLoss, self).__init__()
        self.classifier = classifier
        self.threshold = 0.5  # Initial threshold
        self.momentum = momentum
        self.fairness_lambda = fairness_lambda
        self.temperature = temperature

        # Initialize running averages for normalization
        self.register_buffer("running_ce_avg", torch.tensor(1.0))
        self.register_buffer("running_fairness_avg", torch.tensor(1.0))
        self.running_decay = 0.9  # decay factor for running average updates

    def update_threshold(self, pred_probs, labels):
        """
        Update running threshold using the current batch with momentum.
        
        Args:
            pred_probs: Predicted probabilities [batch_size]
            labels: True labels [batch_size]
        """
        valid_mask = ~torch.isnan(labels) & ~torch.isnan(pred_probs)
        if valid_mask.sum() == 0:
            return
            
        # Detach values for ROC computation
        scores = pred_probs[valid_mask].detach().cpu().numpy()
        y = labels[valid_mask].detach().cpu().numpy()
        
        # Compute ROC and find threshold where sensitivity is closest to specificity
        fpr, sens, threshs = roc_curve(y, scores)
        spec = 1 - fpr
        new_threshold = threshs[np.argmin(np.abs(spec - sens))]
        
        # Update running threshold with momentum
        self.threshold = (1 - self.momentum) * self.threshold + self.momentum * new_threshold

    def calculate_group_rates(self, pred_probs, labels, attr_values, attr):
        """
        Calculate prediction rates for each group using a differentiable soft threshold.
        
        Args:
            pred_probs: Predicted probabilities [batch_size]
            labels: True labels [batch_size]
            attr_values: Unique values in the protected attribute
            attr: Protected attribute values [batch_size]
            
        Returns:
            List of dictionaries with TPR and FPR for each group.
        """
        group_rates = []
        
        # Use a sigmoid with a temperature to create differentiable "soft" predictions
        predictions = torch.sigmoid((pred_probs - self.threshold) / self.temperature)
        
        # Create a valid mask to filter out invalid entries
        attr = attr.unsqueeze(1)
        valid_mask = ~(torch.isnan(labels) | torch.isnan(pred_probs) | torch.isnan(attr) | 
                       (labels == -1) | (pred_probs == -1) | (attr == -1))
        
        for value in attr_values:
            # Mask for the current group
            group_mask = (attr == value) & valid_mask
            pos_mask = group_mask & (labels == 1)
            neg_mask = group_mask & (labels == 0)
            
            if not group_mask.any():
                continue
                
            # Compute rates using the soft predictions
            tpr = predictions[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_probs.device)
            fpr = predictions[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_probs.device)
            
            group_rates.append({'tpr': tpr, 'fpr': fpr})
            
        return group_rates

    def calculate_eodds(self, pred_probs, labels, protected_attrs):
        """
        Calculate the Equal Opportunity Difference (EOODs) for multiple protected attributes.
        
        Args:
            pred_probs: Predicted probabilities from classifier [batch_size, num_classes]
            labels: True labels [batch_size, num_classes]
            protected_attrs: Protected attributes [batch_size, num_attributes] (e.g., sex, age, gender)
            
        Returns:
            Maximum EOODs value across all protected attributes.
        """
        pred_probs = pred_probs.float()
        labels = labels.float()
        protected_attrs = protected_attrs.float()
        
        max_eodds = torch.tensor(0.0, device=pred_probs.device)
        
        # Loop over each protected attribute
        for attr_idx in range(protected_attrs.size(1)):
            attr = protected_attrs[:, attr_idx]
            attr_values = torch.unique(attr[~torch.isnan(attr)])
            
            if len(attr_values) < 2:
                continue

            # Loop over each class (for multi-class tasks)
            for class_idx in range(pred_probs.size(1)):
                class_pred_probs = pred_probs[:, class_idx].unsqueeze(1)
                class_labels = labels[:, class_idx].unsqueeze(1)
                
                # Compute group rates using the soft thresholding mechanism
                group_rates = self.calculate_group_rates(class_pred_probs, class_labels, attr_values, attr)
            
                if len(group_rates) < 2:
                    continue
                    
                # Extract TPR and FPR for each group
                tpr_values = torch.stack([rates['tpr'] for rates in group_rates])
                fpr_values = torch.stack([rates['fpr'] for rates in group_rates])
                
                tpr_diff = torch.max(tpr_values) - torch.min(tpr_values)
                fpr_diff = torch.max(fpr_values) - torch.min(fpr_values)
                
                # Equalized odds difference: average of TPR and FPR differences
                eodds = (tpr_diff + fpr_diff) / 2
                max_eodds = torch.maximum(max_eodds, eodds)
        
        return max_eodds
    
    def calculate_group_ce_loss(self, pred_probs, labels, protected_attrs):
        """
        For each protected attribute and each class, compute the binary cross entropy (BCE)
        loss per group and then penalize differences across groups (using a max-min difference).
        """
        total_loss = 0.0
        count = 0
        # Loop over each protected attribute.
        for attr_idx in range(protected_attrs.size(1)):
            attr = protected_attrs[:, attr_idx]
            attr_values = torch.unique(attr[~torch.isnan(attr)])
            if len(attr_values) < 2:
                continue
            # Loop over each class.
            for class_idx in range(pred_probs.size(1)):
                group_losses = []
                for value in attr_values:
                    group_mask = (attr == value)
                    valid_mask = group_mask & ~torch.isnan(labels[:, class_idx])
                    if valid_mask.sum() > 0:
                        loss_val = F.binary_cross_entropy(
                            pred_probs[valid_mask, class_idx],
                            labels[valid_mask, class_idx]
                        )
                        group_losses.append(loss_val)
                if len(group_losses) >= 2:
                    group_losses_tensor = torch.stack(group_losses)
                    diff_loss = torch.max(group_losses_tensor) - torch.min(group_losses_tensor)
                    total_loss += diff_loss
                    count += 1
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=pred_probs.device)

    
    def forward(self, reconstructed_images, labels, protected_attrs):
        # Compute classifier predictions on reconstructed images
        pred_probs = torch.sigmoid(self.classifier(reconstructed_images))
        
        # For multi-label classification, compute CE loss only on non-NaN labels
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() == 0:
            ce_loss = torch.tensor(0.0, device=pred_probs.device)
        else:
            # Mask out NaN values for BCE loss
            ce_loss = F.cross_entropy(
                pred_probs[valid_mask],
                labels[valid_mask],
                reduction='mean'
            )
            #ce_loss = self.calculate_group_ce_loss(pred_probs, labels, protected_attrs)
        
        # Update threshold using only valid predictions
        self.update_threshold(pred_probs[valid_mask], labels[valid_mask])
        
        # Compute fairness loss based on equalized odds difference
        eodds = self.calculate_eodds(pred_probs, labels, protected_attrs)
        fairness_loss = eodds ** 2

        # Update running averages and normalize losses
        with torch.no_grad():
            self.running_ce_avg = self.running_decay * self.running_ce_avg + (1 - self.running_decay) * ce_loss.detach()
            self.running_fairness_avg = self.running_decay * self.running_fairness_avg + (1 - self.running_decay) * fairness_loss.detach()

        # Normalize the losses
        eps = 1e-6
        norm_ce_loss = ce_loss / (self.running_ce_avg + eps)
        norm_fairness_loss = fairness_loss / (self.running_fairness_avg + eps)

        # Combine the scaled losses
        total_loss = self.fairness_lambda * (norm_fairness_loss + norm_ce_loss)
        return total_loss