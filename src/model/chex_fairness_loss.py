import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
import numpy as np

class ChexFairnessLoss(nn.Module):
    def __init__(self, classifier, momentum=0.1):
        """
        Initialize the fairness loss module.
        
        Args:
            classifier: Pre-trained classifier model
            momentum: Momentum for updating running threshold (default: 0.1)
        """
        super(ChexFairnessLoss, self).__init__()
        self.classifier = classifier
        self.threshold = 0.5  # Initial threshold
        self.momentum = momentum

    def update_threshold(self, pred_probs, labels):
        """
        Update running threshold using current batch with momentum.
        
        Args:
            pred_probs: Predicted probabilities [batch_size]
            labels: True labels [batch_size]
        """
        # Convert to numpy for sklearn compatibility
        valid_mask = ~torch.isnan(labels) & ~torch.isnan(pred_probs)
        if valid_mask.sum() == 0:
            return
            
        scores = pred_probs[valid_mask].cpu().numpy()
        y = labels[valid_mask].cpu().numpy()
        
        # Compute new threshold
        fpr, sens, threshs = roc_curve(y, scores)
        spec = 1 - fpr
        new_threshold = threshs[np.argmin(np.abs(spec - sens))]
        
        # Update running threshold with momentum
        self.threshold = (1 - self.momentum) * self.threshold + self.momentum * new_threshold
    
    def calculate_group_rates(self, pred_probs, labels, attr_values, attr):
        """
        Calculate prediction rates for each group.
        
        Args:
            pred_probs: Predicted probabilities [batch_size]
            labels: True labels [batch_size]
            attr_values: Unique values in the protected attribute
            attr: Protected attribute values [batch_size]
            
        Returns:
            Dictionary of rates for each group
        """
        group_rates = []
        
        # Binary predictions using current threshold
        predictions = (pred_probs >= self.threshold).float()
        
        # Create valid mask for the entire batch
        attr = attr.unsqueeze(1)
        valid_mask = ~(torch.isnan(labels) | torch.isnan(pred_probs) | torch.isnan(attr) | 
                      (labels == -1) | (pred_probs == -1) | (attr == -1))
        
        for value in attr_values:
            # Calculate masks for this group
            group_mask = (attr == value) & valid_mask
            pos_mask = group_mask & (labels == 1)
            neg_mask = group_mask & (labels == 0)
            
            # Skip if no samples in this group
            if not group_mask.any():
                continue
                
            # Calculate rates
            tpr = predictions[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_probs.device)
            fpr = predictions[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_probs.device)
            
            group_rates.append({'tpr': tpr, 'fpr': fpr})
            
        return group_rates
    
    def calculate_eodds(self, pred_probs, labels, protected_attrs):
        """
        Calculate the Equal Opportunity Difference (EOODs) for multiple protected attributes
        
        Args:
            pred_probs: Predicted probabilities from classifier [batch_size]
            labels: True labels [batch_size]
            protected_attrs: Protected attributes [batch_size, 3] for sex, age, gender
            
        Returns:
            max EOODs value across all protected attributes
        """
        # Convert inputs to float for stable division
        pred_probs = pred_probs.float()
        labels = labels.float()
        protected_attrs = protected_attrs.float()
        
        max_eodds = torch.tensor(0.0, device=pred_probs.device)
        
        # Calculate EOODs for each protected attribute
        for attr_idx in range(protected_attrs.size(1)):
            attr = protected_attrs[:, attr_idx]
            attr_values = torch.unique(attr[~torch.isnan(attr)])
            
            if len(attr_values) < 2:
                continue

            for class_idx in range(pred_probs.size(1)):
                class_pred_probs = pred_probs[:, class_idx].unsqueeze(1)
                class_labels = labels[:, class_idx].unsqueeze(1)
                
                # Get rates for each group
                group_rates = self.calculate_group_rates(class_pred_probs, class_labels, attr_values, attr)
            
                if len(group_rates) < 2:
                    continue
                    
                # Calculate max difference in TPR and FPR across all pairs
                tpr_values = torch.tensor([rates['tpr'] for rates in group_rates])
                fpr_values = torch.tensor([rates['fpr'] for rates in group_rates])
                
                tpr_diff = torch.max(tpr_values) - torch.min(tpr_values)
                fpr_diff = torch.max(fpr_values) - torch.min(fpr_values)
                
                # EOODs is average of TPR difference and FPR difference
                eodds = (tpr_diff + fpr_diff) / 2
                max_eodds = torch.maximum(max_eodds, eodds)
        
        return max_eodds
    
    def forward(self, images, labels, protected_attrs):
        """
        Calculate the fairness loss for a batch.
        
        Args:
            images: Batch of input images
            labels: Ground truth labels [batch_size]
            protected_attrs: Protected attributes [batch_size, 3]
            
        Returns:
            Squared max EOODs loss
        """
        with torch.no_grad():
            pred_probs = torch.sigmoid(self.classifier(images))
        
        # Update threshold using whole batch
        self.update_threshold(pred_probs, labels)
        
        eodds = self.calculate_eodds(pred_probs, labels, protected_attrs)
        loss = eodds ** 2
        
        return loss