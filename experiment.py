
#!/usr/bin/env python3
"""
Robust Conformal Prediction under Lévy-Prokhorov Distribution Shifts
Implementation based on the paper: "Conformal Prediction under Lévy-Prokhorov Distribution Shifts: 
Robustness to Local and Global Perturbations"

This script implements a simplified version of LP-robust conformal prediction for demonstration purposes.
It focuses on the core algorithmic components while maintaining compatibility with serverless environments.
"""

import sys
import numpy as np
import logging
from typing import Tuple, List, Optional
from scipy.stats import norm, rankdata
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class LPRobustConformal:
    """
    Implementation of Lévy-Prokhorov robust conformal prediction.
    
    This class implements the core algorithm for constructing prediction intervals
    that are robust to distribution shifts modeled via LP ambiguity sets.
    """
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, rho: float = 0.05):
        """
        Initialize the LP robust conformal predictor.
        
        Args:
            alpha: Desired miscoverage level (1 - coverage)
            epsilon: Local perturbation parameter for LP ambiguity set
            rho: Global perturbation parameter for LP ambiguity set
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.quantile_threshold = None
        self.is_fitted = False
        
        logger.info(f"Initialized LP Robust Conformal Predictor with alpha={alpha}, epsilon={epsilon}, rho={rho}")
    
    def _nonconformity_score(self, probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores using the negative log-likelihood.
        
        Args:
            probabilities: Predicted probabilities for each class
            labels: True labels
            
        Returns:
            Array of nonconformity scores
        """
        try:
            # Negative log-likelihood score
            n_samples = len(labels)
            scores = -np.log(probabilities[np.arange(n_samples), labels] + 1e-10)
            return scores
        except Exception as e:
            logger.error(f"Error computing nonconformity scores: {e}")
            raise
    
    def _worst_case_quantile(self, scores: np.ndarray) -> float:
        """
        Compute worst-case quantile under LP distribution shifts.
        
        Args:
            scores: Nonconformity scores from calibration set
            
        Returns:
            Worst-case quantile value
        """
        try:
            n_calib = len(scores)
            level_adjusted = (1.0 - self.alpha) * (1.0 + 1.0 / n_calib)
            
            # Apply LP robustness adjustment
            robust_level = min(level_adjusted + self.rho, 1.0)
            base_quantile = np.quantile(scores, robust_level)
            worst_case_quantile = base_quantile + self.epsilon
            
            logger.info(f"Computed worst-case quantile: {worst_case_quantile:.4f} "
                       f"(base: {base_quantile:.4f}, robust_level: {robust_level:.4f})")
            
            return worst_case_quantile
        except Exception as e:
            logger.error(f"Error computing worst-case quantile: {e}")
            raise
    
    def fit(self, calibration_scores: np.ndarray) -> None:
        """
        Fit the conformal predictor using calibration scores.
        
        Args:
            calibration_scores: Nonconformity scores from calibration set
        """
        try:
            logger.info(f"Fitting LP robust conformal predictor with {len(calibration_scores)} calibration scores")
            
            if len(calibration_scores) == 0:
                raise ValueError("Calibration scores cannot be empty")
            
            self.quantile_threshold = self._worst_case_quantile(calibration_scores)
            self.is_fitted = True
            
            logger.info(f"Successfully fitted model with quantile threshold: {self.quantile_threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting conformal predictor: {e}")
            sys.exit(1)
    
    def predict_sets(self, test_probabilities: np.ndarray) -> List[List[int]]:
        """
        Generate prediction sets for test instances.
        
        Args:
            test_probabilities: Predicted probabilities for test instances
            
        Returns:
            List of prediction sets (each set contains class indices)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            n_test, n_classes = test_probabilities.shape
            prediction_sets = []
            
            for i in range(n_test):
                # For each class, compute the nonconformity score
                class_scores = []
                for j in range(n_classes):
                    # Create artificial label j for this test point
                    score = -np.log(test_probabilities[i, j] + 1e-10)
                    class_scores.append(score)
                
                # Include classes with scores below the threshold
                prediction_set = [j for j, score in enumerate(class_scores) 
                               if score <= self.quantile_threshold]
                prediction_sets.append(prediction_set)
            
            logger.info(f"Generated prediction sets for {n_test} test instances")
            return prediction_sets
            
        except Exception as e:
            logger.error(f"Error generating prediction sets: {e}")
            raise

def evaluate_predictions(prediction_sets: List[List[int]], true_labels: np.ndarray, 
                        alpha: float) -> dict:
    """
    Evaluate the performance of conformal prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True labels
        alpha: Target miscoverage level
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        n_samples = len(true_labels)
        
        # Compute coverage
        coverage = np.mean([true_labels[i] in prediction_sets[i] for i in range(n_samples)])
        
        # Compute average set size
        avg_set_size = np.mean([len(s) for s in prediction_sets])
        
        # Compute conditional coverage (coverage when prediction set is non-empty)
        non_empty_indices = [i for i, s in enumerate(prediction_sets) if len(s) > 0]
        if non_empty_indices:
            conditional_coverage = np.mean([true_labels[i] in prediction_sets[i] 
                                          for i in non_empty_indices])
        else:
            conditional_coverage = 0.0
        
        metrics = {
            'marginal_coverage': coverage,
            'target_coverage': 1 - alpha,
            'average_set_size': avg_set_size,
            'conditional_coverage': conditional_coverage,
            'efficiency': 1.0 / (avg_set_size + 1e-10)  # Inverse of set size
        }
        
        logger.info(f"Evaluation results - Coverage: {coverage:.4f} (target: {1-alpha:.4f}), "
                   f"Avg set size: {avg_set_size:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating predictions: {e}")
        raise

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20, 
                          n_classes: int = 3, test_size: float = 0.3) -> Tuple:
    """
    Generate synthetic classification data for demonstration.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_classes: Number of classes
        test_size: Proportion of test data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features, {n_classes} classes")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data generated - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        sys.exit(1)

def main():
    """
    Main experiment function demonstrating LP-robust conformal prediction.
    """
    logger.info("Starting LP-Robust Conformal Prediction Experiment")
    
    try:
        # Experiment parameters
        ALPHA = 0.1  # 90% coverage target
        EPSILON = 0.1  # Local perturbation parameter
        RHO = 0.05  # Global perturbation parameter
        N_SPLITS = 5  # Number of random splits for evaluation
        
        # Generate synthetic data
        X_train, X_test, y_train, y_test = generate_synthetic_data()
        
        # Train a base classifier
        logger.info("Training base classifier...")
        base_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        base_classifier.fit(X_train, y_train)
        
        # Evaluate base classifier accuracy
        train_accuracy = accuracy_score(y_train, base_classifier.predict(X_train))
        test_accuracy = accuracy_score(y_test, base_classifier.predict(X_test))
        logger.info(f"Base classifier accuracy - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
        
        # Get predicted probabilities
        train_probs = base_classifier.predict_proba(X_train)
        test_probs = base_classifier.predict_proba(X_test)
        
        # Initialize results storage
        all_metrics = []
        
        # Run multiple splits for robust evaluation
        logger.info(f"Running {N_SPLITS} random splits for evaluation...")
        
        for split in range(N_SPLITS):
            logger.info(f"Split {split + 1}/{N_SPLITS}")
            
            try:
                # Split data into calibration and proper test set
                calib_indices, test_indices = train_test_split(
                    np.arange(len(X_test)), test_size=0.5, random_state=split
                )
                
                X_calib = X_test[calib_indices]
                y_calib = y_test[calib_indices]
                calib_probs = test_probs[calib_indices]
                
                X_proper_test = X_test[test_indices]
                y_proper_test = y_test[test_indices]
                proper_test_probs = test_probs[test_indices]
                
                # Compute nonconformity scores for calibration set
                conformal_predictor = LPRobustConformal(alpha=ALPHA, epsilon=EPSILON, rho=RHO)
                calib_scores = conformal_predictor._nonconformity_score(calib_probs, y_calib)
                
                # Fit the conformal predictor
                conformal_predictor.fit(calib_scores)
                
                # Generate prediction sets for test data
                prediction_sets = conformal_predictor.predict_sets(proper_test_probs)
                
                # Evaluate predictions
                metrics = evaluate_predictions(prediction_sets, y_proper_test, ALPHA)
                metrics['split'] = split + 1
                all_metrics.append(metrics)
                
                logger.info(f"Split {split + 1} completed - Coverage: {metrics['marginal_coverage']:.4f}")
                
            except Exception as e:
                logger.warning(f"Split {split + 1} failed: {e}")
                continue
        
        # Compute average metrics across splits
        if all_metrics:
            avg_coverage = np.mean([m['marginal_coverage'] for m in all_metrics])
            avg_set_size = np.mean([m['average_set_size'] for m in all_metrics])
            avg_efficiency = np.mean([m['efficiency'] for m in all_metrics])
            
            logger.info("\n" + "="*50)
            logger.info("FINAL EXPERIMENT RESULTS")
            logger.info("="*50)
            logger.info(f"Target coverage level: {1 - ALPHA:.4f}")
            logger.info(f"Average empirical coverage: {avg_coverage:.4f}")
            logger.info(f"Average prediction set size: {avg_set_size:.4f}")
            logger.info(f"Average efficiency: {avg_efficiency:.4f}")
            logger.info(f"LP robustness parameters - epsilon: {EPSILON}, rho: {RHO}")
            logger.info(f"Successful splits: {len(all_metrics)}/{N_SPLITS}")
            
            # Check if coverage guarantee is maintained
            coverage_diff = avg_coverage - (1 - ALPHA)
            if coverage_diff >= -0.02:  # Allow small tolerance
                logger.info("✓ Coverage guarantee maintained")
            else:
                logger.warning("⚠ Coverage below target level")
                
        else:
            logger.error("No successful splits completed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
