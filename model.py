"""
XGBoost Model for Credit Card Delinquency Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
import joblib
import json
from datetime import datetime


class DelinquencyPredictor:
    """
    XGBoost-based delinquency prediction model
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.performance_metrics = {}
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              hyperparameter_tuning=True):
        """
        Train XGBoost model with optional hyperparameter tuning
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation labels
        hyperparameter_tuning : bool
            Whether to perform GridSearchCV
        """
        
        print("Training XGBoost model...")
        self.feature_names = X_train.columns.tolist()
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning (this may take a few minutes)...")
            
            # Parameter grid for GridSearch
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            # Base model
            base_model = xgb.XGBClassifier(
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc'
            )
            
            # GridSearchCV with 5-fold cross-validation
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc'
            )
            
            # Early stopping with validation set
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        
        print("✓ Model training complete!")
        
    def predict(self, X, threshold=0.5):
        """
        Predict delinquency (binary)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        threshold : float
            Classification threshold (default: 0.5)
        
        Returns:
        --------
        np.array
            Binary predictions (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict delinquency probability
        
        Returns:
        --------
        np.array
            Probability of delinquency (0 to 1)
        """
        return self.model.predict_proba(X)[:, 1]
    
    def predict_risk_category(self, X):
        """
        Predict risk category (HIGH/MEDIUM/LOW)
        
        Returns:
        --------
        pd.Series
            Risk categories and scores
        """
        probas = self.predict_proba(X)
        scores = probas * 100
        
        categories = pd.cut(
            scores,
            bins=[0, 30, 60, 100],
            labels=['LOW RISK', 'MEDIUM RISK', 'HIGH RISK'],
            include_lowest=True
        )
        
        return pd.DataFrame({
            'Risk_Score': scores,
            'Risk_Category': categories,
            'Delinquency_Probability': probas
        })
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate model performance on test set
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        threshold : float
            Classification threshold
        
        Returns:
        --------
        dict
            Performance metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Store metrics
        self.performance_metrics = {
            'auc_roc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'threshold': float(threshold)
        }
        
        # Print results
        print(f"\nPerformance Metrics (threshold={threshold}):")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Precision: {precision:.4f} (% of HIGH RISK predictions that are correct)")
        print(f"  Recall:    {recall:.4f} (% of actual delinquencies caught)")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                Non-Del   Delinquent")
        print(f"Actual Non-Del    {tn:4d}      {fp:4d}")
        print(f"Actual Delinq     {fn:4d}      {tp:4d}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Non-Delinquent', 'Delinquent']))
        
        # Business interpretation
        print(f"\nBusiness Impact Interpretation:")
        write_off_cost = 2500
        intervention_cost = 50
        
        prevented_writeoffs = tp * write_off_cost
        false_alarm_cost = fp * intervention_cost
        missed_writeoffs = fn * write_off_cost
        
        baseline_cost = (tp + fn) * write_off_cost
        model_cost = false_alarm_cost + missed_writeoffs
        savings = baseline_cost - model_cost
        
        print(f"  Prevented write-offs: ${prevented_writeoffs:,.0f} ({tp} customers)")
        print(f"  False alarm cost: ${false_alarm_cost:,.0f} ({fp} customers)")
        print(f"  Missed write-offs: ${missed_writeoffs:,.0f} ({fn} customers)")
        print(f"  Total cost savings: ${savings:,.0f} ({savings/baseline_cost*100:.1f}% reduction)")
        
        print("="*60 + "\n")
        
        return self.performance_metrics
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance rankings
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("="*50)
        for idx, row in importance.head(top_n).iterrows():
            print(f"{row['feature']:30s} {row['importance']:.4f}")
        
        return importance
    
    def save_model(self, filepath='models/xgboost_model.pkl'):
        """Save trained model to disk"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved to {filepath}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'features': self.feature_names,
            'best_params': self.best_params,
            'performance_metrics': self.performance_metrics,
            'training_date': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_path}")
    
    def load_model(self, filepath='models/xgboost_model.pkl'):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['features']
        self.best_params = metadata.get('best_params')
        self.performance_metrics = metadata.get('performance_metrics', {})
        
        print(f"✓ Model loaded from {filepath}")
        return self


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Complete training and evaluation pipeline
    """
    # Initialize model
    model = DelinquencyPredictor()
    
    # Train with hyperparameter tuning
    model.train(X_train, y_train, X_val, y_val, hyperparameter_tuning=True)
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test, threshold=0.5)
    
    # Feature importance
    importance_df = model.get_feature_importance(top_n=10)
    
    # Save model
    model.save_model()
    
    return model, metrics, importance_df


if __name__ == "__main__":
    print("XGBoost Delinquency Prediction Model")
    print("Run this module after preprocessing data")
    print("Use notebooks/03_model_training.ipynb for full pipeline")
