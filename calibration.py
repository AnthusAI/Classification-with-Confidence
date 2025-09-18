"""
Confidence Calibration Methods for Language Models

This module provides three main calibration approaches:
1. Platt Scaling (Logistic Regression)
2. Isotonic Regression  
3. Temperature Scaling

Each method transforms raw confidence scores into well-calibrated probabilities
where "X% confident" means "correct X% of the time".
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, Optional
import warnings


class PlattScalingCalibrator:
    """
    Platt Scaling calibration using logistic regression.
    
    Best for:
    - Small validation datasets (100-1000 examples)
    - Sigmoid-shaped calibration curves
    - Fast inference requirements
    
    Limitations:
    - Assumes specific sigmoid functional form
    """
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, labels: np.ndarray, 
            test_size: float = 0.2, random_state: int = 42) -> 'PlattScalingCalibrator':
        """
        Fit Platt scaling calibrator on validation data.
        
        Args:
            confidences: Raw confidence scores from model [0, 1]
            labels: True binary labels (0/1)
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            self: Fitted calibrator
        """
        # Reshape confidences for sklearn
        confidences = confidences.reshape(-1, 1)
        
        # Split data for calibration
        conf_train, conf_val, labels_train, labels_val = train_test_split(
            confidences, labels, test_size=test_size, random_state=random_state
        )
        
        # Fit logistic regression calibrator
        self.calibrator = LogisticRegression()
        self.calibrator.fit(conf_train, labels_train)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, confidences: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw confidence scores.
        
        Args:
            confidences: Raw confidence scores [0, 1]
            
        Returns:
            Calibrated probabilities [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        confidences = confidences.reshape(-1, 1)
        return self.calibrator.predict_proba(confidences)[:, 1]
    
    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            confidence: Raw confidence score [0, 1]
            
        Returns:
            Calibrated probability [0, 1]
        """
        return self.predict_proba(np.array([confidence]))[0]


class IsotonicRegressionCalibrator:
    """
    Isotonic regression calibration (non-parametric).
    
    Best for:
    - Any monotonic calibration relationship
    - No parametric assumptions needed
    - Complex calibration curves
    
    Limitations:
    - Needs more validation data (1000+ examples)
    """
    
    def __init__(self, out_of_bounds: str = 'clip'):
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.is_fitted = False
    
    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> 'IsotonicRegressionCalibrator':
        """
        Fit isotonic regression calibrator.
        
        Args:
            confidences: Raw confidence scores [0, 1]
            labels: True binary labels (0/1)
            
        Returns:
            self: Fitted calibrator
        """
        self.calibrator.fit(confidences, labels)
        self.is_fitted = True
        return self
    
    def predict_proba(self, confidences: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw confidence scores.
        
        Args:
            confidences: Raw confidence scores [0, 1]
            
        Returns:
            Calibrated probabilities [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        return self.calibrator.predict(confidences)
    
    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a single confidence score.
        
        Args:
            confidence: Raw confidence score [0, 1]
            
        Returns:
            Calibrated probability [0, 1]
        """
        return self.predict_proba(np.array([confidence]))[0]


class TemperatureScalingCalibrator(nn.Module):
    """
    Temperature scaling for neural network logits.
    
    Best for:
    - When you have access to model logits
    - Simple, single-parameter method
    - Preserves relative ordering of predictions
    
    Limitations:
    - Only works with neural network outputs
    - Requires access to raw logits, not just probabilities
    """
    
    def __init__(self, initial_temperature: float = 1.5):
        super(TemperatureScalingCalibrator, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
        self.is_fitted = False
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, 
            lr: float = 0.01, max_iter: int = 50) -> 'TemperatureScalingCalibrator':
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: Raw model logits
            labels: True class labels
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            self: Fitted calibrator
        """
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.is_fitted = True
        return self
    
    def get_temperature(self) -> float:
        """Get the learned temperature parameter."""
        return self.temperature.item()
    
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits and return probabilities.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before making predictions")
        
        scaled_logits = self.forward(logits)
        return torch.softmax(scaled_logits, dim=-1)


def collect_calibration_data(model, validation_texts: list, true_labels: list, 
                           confidence_method: str = 'logprobs') -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect confidence scores and correctness labels for calibration.
    
    Args:
        model: Your confidence estimation model
        validation_texts: List of text examples
        true_labels: List of true labels (0/1 for binary classification)
        confidence_method: Method to use ('logprobs', 'consistency', 'combined')
        
    Returns:
        confidences: Array of confidence scores
        accuracies: Array of binary correctness (1 if correct, 0 if wrong)
    """
    confidences = []
    accuracies = []
    
    for text, true_label in zip(validation_texts, true_labels):
        # Get confidence using your preferred method
        # This is a placeholder - replace with your actual confidence function
        if confidence_method == 'logprobs':
            pred, conf = get_logprob_confidence(model, text)
        elif confidence_method == 'consistency':
            pred, conf = get_consistency_confidence(model, text)
        elif confidence_method == 'combined':
            pred, conf = get_combined_confidence(model, text)
        else:
            raise ValueError(f"Unknown confidence method: {confidence_method}")
        
        confidences.append(conf)
        accuracies.append(1 if pred == true_label else 0)
    
    return np.array(confidences), np.array(accuracies)


def get_calibrated_confidence(model, calibrator, text: str, 
                            confidence_method: str = 'logprobs') -> Tuple[str, float]:
    """
    Get calibrated confidence for new text.
    
    Args:
        model: Your confidence estimation model
        calibrator: Fitted calibration model
        text: Input text to classify
        confidence_method: Method to use for raw confidence
        
    Returns:
        prediction: Predicted class
        calibrated_confidence: Calibrated confidence score
    """
    # Get raw confidence
    if confidence_method == 'logprobs':
        prediction, raw_confidence = get_logprob_confidence(model, text)
    elif confidence_method == 'consistency':
        prediction, raw_confidence = get_consistency_confidence(model, text)
    elif confidence_method == 'combined':
        prediction, raw_confidence = get_combined_confidence(model, text)
    else:
        raise ValueError(f"Unknown confidence method: {confidence_method}")
    
    # Apply calibration
    if isinstance(calibrator, TemperatureScalingCalibrator):
        # Temperature scaling works differently - needs logits
        warnings.warn("Temperature scaling requires logits, not confidence scores")
        calibrated_confidence = raw_confidence
    else:
        calibrated_confidence = calibrator.calibrate(raw_confidence)
    
    return prediction, calibrated_confidence


# Placeholder functions - replace with your actual implementations
def get_logprob_confidence(model, text: str) -> Tuple[str, float]:
    """Placeholder for logprob-based confidence estimation."""
    raise NotImplementedError("Replace with your logprob confidence implementation")

def get_consistency_confidence(model, text: str) -> Tuple[str, float]:
    """Placeholder for consistency-based confidence estimation."""
    raise NotImplementedError("Replace with your consistency confidence implementation")

def get_combined_confidence(model, text: str) -> Tuple[str, float]:
    """Placeholder for combined confidence estimation."""
    raise NotImplementedError("Replace with your combined confidence implementation")


if __name__ == "__main__":
    # Example usage
    print("Calibration methods available:")
    print("1. PlattScalingCalibrator - for small datasets, sigmoid curves")
    print("2. IsotonicRegressionCalibrator - non-parametric, flexible")
    print("3. TemperatureScalingCalibrator - for neural network logits")
    
    # Example with synthetic data
    np.random.seed(42)
    raw_confidences = np.random.beta(2, 2, 1000)  # Simulated confidence scores
    true_accuracies = (raw_confidences > 0.5).astype(int)  # Simulated correctness
    
    # Fit Platt scaling
    platt_cal = PlattScalingCalibrator()
    platt_cal.fit(raw_confidences, true_accuracies)
    
    # Test calibration
    test_confidence = 0.85
    calibrated = platt_cal.calibrate(test_confidence)
    print(f"\nExample: Raw confidence {test_confidence:.2f} → Calibrated {calibrated:.2f}")
