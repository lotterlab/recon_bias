import torch
import pytest
from src.model.chex_fairness_loss import ChexFairnessLoss

class MockClassifier(torch.nn.Module):
    def forward(self, x):
        return x

def test_calculate_group_rates_binary_correct():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    # Create test data - shape: (batch_size, 1)
    pred_probs = torch.tensor([0.8, 0.3, 0.7, 0.2, 0.9, 0.1]).unsqueeze(1)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
    attr = torch.tensor([0, 0, 0, 1, 1, 1])  # Two groups
    attr_values = torch.tensor([0, 1])
    
    # Calculate group rates
    group_rates = fairness_loss.calculate_group_rates(pred_probs, labels, attr_values, attr)
    
    # Check results
    assert len(group_rates) == 2
    # Group 0: 2 positive samples (0.8, 0.7), 1 negative sample (0.3)
    assert abs(group_rates[0]['tpr'].item() - 1.0) < 1e-6  # Both positives > 0.5
    assert abs(group_rates[0]['fpr'].item() - 0.0) < 1e-6  # Negative < 0.5

    # Group 1: 1 positive sample (0.9), 2 negative samples (0.2, 0.1)
    assert abs(group_rates[1]['tpr'].item() - 1.0) < 1e-6  # Positive > 0.5
    assert abs(group_rates[1]['fpr'].item() - 0.0) < 1e-6  # Both negatives < 0.5


def test_calculate_group_rates_binary_incorrect():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    # Create test data - shape: (batch_size, 1)
    pred_probs = torch.tensor([0.3, 1, 0.7, 0.2, 0.4, 0.1]).unsqueeze(1)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
    attr = torch.tensor([0, 0, 0, 1, 1, 1])  # Two groups
    attr_values = torch.tensor([0, 1])
    
    # Calculate group rates
    group_rates = fairness_loss.calculate_group_rates(pred_probs, labels, attr_values, attr)
    
    # Check results
    assert len(group_rates) == 2
    # Group 0: 2 positive samples (0.8, 0.7), 1 negative sample (0.3)
    assert abs(group_rates[0]['tpr'].item() - 1/2) < 1e-6  # Both positives > 0.5
    assert abs(group_rates[0]['fpr'].item() - 1) < 1e-6  # Negative < 0.5

    # Group 1: 1 positive sample (0.9), 2 negative samples (0.2, 0.1)
    assert abs(group_rates[1]['tpr'].item() - 0.0) < 1e-6  # Positive > 0.5
    assert abs(group_rates[1]['fpr'].item() - 0.0) < 1e-6  # Both negatives < 0.5

def test_calculate_group_rates_multiclass():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    # Create test data for 3 classes
    # Shape: (batch_size, num_classes)
    pred_probs = torch.tensor([
        [0.8, 0.1, 0.1],  # pred class 0
        [0.1, 0.8, 0.1],  # pred class 1
        [0.1, 0.1, 0.8],  # pred class 2
        [0.7, 0.2, 0.1],  # pred class 0
        [0.1, 0.7, 0.2],  # pred class 1
        [0.2, 0.1, 0.7]   # pred class 2
    ])
    
    # One-hot encoded labels
    labels = torch.tensor([
        [1.0, 0.0, 0.0],  # true class 0
        [0.0, 1.0, 0.0],  # true class 1
        [0.0, 0.0, 1.0],  # true class 2
        [1.0, 0.0, 0.0],  # true class 0
        [0.0, 1.0, 0.0],  # true class 1
        [0.0, 0.0, 1.0]   # true class 2
    ])
    
    attr = torch.tensor([0, 0, 0, 1, 1, 1])
    attr_values = torch.tensor([0, 1])
    
    # Calculate group rates for each class
    for class_idx in range(3):
        class_pred_probs = pred_probs[:, class_idx].unsqueeze(1)
        class_labels = labels[:, class_idx].unsqueeze(1)
        
        group_rates = fairness_loss.calculate_group_rates(
            class_pred_probs, class_labels, attr_values, attr
        )
        
        # Both groups should have perfect prediction
        assert len(group_rates) == 2
        for group_idx in range(2):
            if torch.any(class_labels[attr == group_idx] == 1):
                assert abs(group_rates[group_idx]['tpr'].item() - 1.0) < 1e-6
            if torch.any(class_labels[attr == group_idx] == 0):
                assert abs(group_rates[group_idx]['fpr'].item() - 0.0) < 1e-6

def test_calculate_eodds():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    # Create test data with batched dimensions
    pred_probs = torch.tensor([0.9, 0.1, 0.9, 0.8, 0.7, 0.9]).unsqueeze(1)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unsqueeze(1)
    protected_attrs = torch.tensor([0, 0, 0, 1, 1, 1]).unsqueeze(1)
    
    eodds = fairness_loss.calculate_eodds(pred_probs, labels, protected_attrs)
    # Group 0: TPR = 1.0, FPR = 0.0
    # Group 1: TPR = 1.0, FPR = 1.0
    # Expected EODDS = (|1.0 - 1.0| + |0.0 - 1.0|) / 2 = 0.5
    assert abs(eodds.item() - 0.5) < 1e-6

def test_calculate_eodds_multiattr():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    pred_probs = torch.tensor(
        [0.4, 0.1, 1.0, 0.7, 0.1, 0.2],  # pred class 0
    ).unsqueeze(1)
    
    # One-hot encoded labels
    labels = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ).unsqueeze(1)
    
    protected_attrs = torch.tensor([
        [0, 0, 0],  # protected attr 0
        [0, 1, 0],  # protected attr 0
        [0, 0, 1],  # protected attr 0
        [1, 1, 1],  # protected attr 1
        [1, 1, 0],  # protected attr 1
        [1, 1, 0]   # protected attr 1
    ])
    
    eodds = fairness_loss.calculate_eodds(pred_probs, labels, protected_attrs)
    print(eodds)

    assert abs(eodds.item() - 1) < 1e-6


def test_calculate_eodds_multiattr2():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    pred_probs = torch.tensor(
        [0.4, 0.6, 1.0, 0.7, 0.1, 0.2],  # pred class 0
    ).unsqueeze(1)
    
    # One-hot encoded labels
    labels = torch.tensor(
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ).unsqueeze(1)
    
    protected_attrs = torch.tensor([
        [0, 0],  # protected attr 0
        [0, 0],  # protected attr 0
        [0, 1],  # protected attr 0
        [1, 1],  # protected attr 1
        [1, 1],  # protected attr 1
        [1, 1]   # protected attr 1
    ])
    
    eodds = fairness_loss.calculate_eodds(pred_probs, labels, protected_attrs)

    assert abs(eodds.item() - 0.75) < 1e-6


def test_handle_missing_values():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    pred_probs = torch.tensor([0.8, 0.3, float('nan'), 0.2, 0.9, 0.1]).unsqueeze(1)
    labels = torch.tensor([1.0, 0.0, 1.0, float('nan'), 1.0, 0.0]).unsqueeze(1)
    attr = torch.tensor([0, 0, 0, 1, 1, 1])
    attr_values = torch.tensor([0, 1])
    
    group_rates = fairness_loss.calculate_group_rates(pred_probs, labels, attr_values, attr)
    
    assert len(group_rates) == 2
    assert not torch.isnan(group_rates[0]['tpr'])
    assert not torch.isnan(group_rates[0]['fpr'])
    assert not torch.isnan(group_rates[1]['tpr'])
    assert not torch.isnan(group_rates[1]['fpr'])

def test_forward_zero_eodds():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    pred_probs = torch.tensor([0.9, 0.1, 0.9, 0.8, 0.2, 0.9]).unsqueeze(1)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unsqueeze(1)
    protected_attrs = torch.tensor([0, 0, 0, 1, 1, 1]).unsqueeze(1)
    
    loss = fairness_loss.forward(pred_probs, labels, protected_attrs)
    
    # Loss should be the square of EODDS (0.5^2 = 0.25)
    assert abs(loss.item() - 0.0) < 1e-6 

def test_forward_non_zero_eodds():
    fairness_loss = ChexFairnessLoss(MockClassifier())
    
    pred_probs = torch.tensor(
        [0.4, 0.6, 1.0, 0.7, 0.1, 0.2],  # pred class 0
    ).unsqueeze(1)
    
    # One-hot encoded labels
    labels = torch.tensor(
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ).unsqueeze(1)
    pred_probs = torch.log(pred_probs / (1 - pred_probs))
    
    protected_attrs = torch.tensor([0, 0, 0, 1, 1, 1]).unsqueeze(1)
    
    loss = fairness_loss.forward(pred_probs, labels, protected_attrs)    
    # Loss should be the square of EODDS (0.75^2 = 0.5625)
    assert abs(loss.item() - 0.5625) < 1e-6 