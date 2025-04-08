import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(BiasPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

class FairnessLoss(nn.Module):
    def __init__(self, classifier, momentum=0.1, device=None, learning_rate=1e-4):
        super(FairnessLoss, self).__init__()
        self.classifier = classifier
        self.momentum = momentum
        self.device = device
        
        # Feature extractor (frozen)
        self.feature_extractor = nn.Sequential(*list(classifier.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Initialize bias predictor to None - we'll create it with the correct dimensions on first forward pass
        self.bias_predictor = None
        
        # Add optimizer - will be initialized when bias_predictor is created
        self.optimizer = None
        self.learning_rate = learning_rate

        # Initialize running averages for normalization
        self.register_buffer("running_ce_avg", torch.tensor(1.0))
        self.register_buffer("running_adv_avg", torch.tensor(1.0))
        self.running_decay = 0.9
        self.eps = 1e-8

    def correlation_loss(self, predictions, targets):
        # Ensure inputs are valid (no NaN or Inf)
        if torch.isnan(predictions).any() or torch.isnan(targets).any() or \
           torch.isinf(predictions).any() or torch.isinf(targets).any():
            return torch.tensor(0.0, device=predictions.device)
            
        vx = predictions - predictions.mean()
        vy = targets - targets.mean()
        
        # Check for zero division
        if torch.sum(vx ** 2).item() < self.eps or torch.sum(vy ** 2).item() < self.eps:
            return torch.tensor(0.0, device=predictions.device)
            
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)) + self.eps)
        return corr ** 2

    def forward(self, reconstructed_images, labels, protected_attrs):
        # Extract features (frozen)
        features = self.feature_extractor(reconstructed_images)
            
        # Flatten the features to handle various feature shapes
        if features.dim() > 2:
            features = features.reshape(features.size(0), -1)
        
        # Initialize bias predictor and its optimizer if not already created
        if self.bias_predictor is None:
            feature_dim = features.size(1)
            self.bias_predictor = BiasPredictor(feature_dim)
            if self.device is not None:
                self.bias_predictor.to(self.device)
            self.optimizer = torch.optim.Adam(self.bias_predictor.parameters(), lr=self.learning_rate)
            print(f"Initialized BiasPredictor with input dimension: {feature_dim}")

        # Classification loss
        pred_logits = self.classifier(reconstructed_images)
        pred_probs = torch.sigmoid(pred_logits)
        
        # Ensure probabilities are in valid range for BCE
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1-1e-7)

        # Create a valid mask for labels
        valid_labels_mask = ~torch.isnan(labels)
        if not valid_labels_mask.any():  # No valid labels
            ce_loss = torch.tensor(0.0, device=pred_probs.device)
        else:
            valid_labels = torch.clamp(labels[valid_labels_mask], min=0.0, max=1.0)
            valid_preds = pred_probs[valid_labels_mask]
            
            ce_loss = F.binary_cross_entropy(valid_preds, valid_labels, reduction='mean')

        # Handle protected attributes - ensure they're valid
        if protected_attrs is None or protected_attrs.size(0) == 0:
            adv_loss = torch.tensor(0.0, device=features.device)
            adv_loss_fe = torch.tensor(0.0, device=features.device)
        else:
            # Create valid mask for protected attributes - check ALL dimensions for NaN
            # First check if it's multi-dimensional
            if protected_attrs.dim() > 1 and protected_attrs.size(1) > 1:
                # Check across all columns by creating a mask that requires all columns to be valid
                valid_attr_mask = ~torch.isnan(protected_attrs).any(dim=1)
            else:
                # For single column, just check that column
                valid_attr_mask = ~torch.isnan(protected_attrs.reshape(-1))
            
            if not valid_attr_mask.any():  # No valid protected attributes
                adv_loss = torch.tensor(0.0, device=features.device)
                adv_loss_fe = torch.tensor(0.0, device=features.device)
            else:
                # Apply bias predictor on all features first
                bias_pred = self.bias_predictor(features.detach())
                bias_pred_fe = self.bias_predictor(features)
                
                # Filter out valid entries for loss computation
                valid_bias_pred = bias_pred[valid_attr_mask]
                valid_bias_pred_fe = bias_pred_fe[valid_attr_mask]
                
                # Use first column for correlation calculation, but only from rows that are fully valid
                valid_protected = protected_attrs[valid_attr_mask, 0]
                
                # Compute adversarial losses
                adv_loss = self.correlation_loss(valid_bias_pred, valid_protected)
                adv_loss_fe = -self.correlation_loss(valid_bias_pred_fe, valid_protected)

        # Update running averages
        self.running_ce_avg = self.running_decay * self.running_ce_avg + (1 - self.running_decay) * ce_loss.detach()
        self.running_adv_avg = self.running_decay * self.running_adv_avg + (1 - self.running_decay) * adv_loss.detach()

        # Normalize losses
        norm_ce_loss = ce_loss / (self.running_ce_avg + self.eps)
        norm_adv_loss_fe = adv_loss_fe / (self.running_adv_avg + self.eps)
        norm_adv_loss = adv_loss / (self.running_adv_avg + self.eps)
        
        # Update bias predictor
        if protected_attrs is not None and protected_attrs.size(0) > 0:
            self.optimizer.zero_grad()
            norm_adv_loss.backward(retain_graph=True)  # Train bias predictor to detect bias
            self.optimizer.step()

        # Return loss for reconstruction model
        return norm_ce_loss + norm_adv_loss_fe  # Only return losses relevant to reconstruction