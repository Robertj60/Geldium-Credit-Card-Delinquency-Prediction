"""
Fairness Testing and Bias Detection for Credit Risk Model
Implements EEOC 4/5ths rule and other fairness metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class FairnessAuditor:
    """
    Comprehensive fairness testing for ML models
    """
    
    def __init__(self, threshold_lower=0.80, threshold_upper=1.25):
        """
        Initialize fairness auditor
        
        Parameters:
        -----------
        threshold_lower : float
            Lower bound for adverse impact ratio (EEOC 4/5ths = 0.80)
        threshold_upper : float
            Upper bound for adverse impact ratio
        """
        self.threshold_lower = threshold_lower
        self.threshold_upper = threshold_upper
        self.audit_results = {}
        
    def adverse_impact_ratio(self, y_pred, protected_attribute, 
                            reference_group=None, protected_group=None):
        """
        Calculate Adverse Impact Ratio (EEOC 4/5ths Rule)
        
        Parameters:
        -----------
        y_pred : array-like
            Model predictions (0 or 1, or HIGH RISK classifications)
        protected_attribute : array-like
            Demographic attribute (e.g., age group, location)
        reference_group : str, optional
            Reference group name (if None, uses group with highest selection rate)
        protected_group : str, optional
            Protected group name (if None, tests all groups)
        
        Returns:
        --------
        pd.DataFrame
            Adverse impact ratios for all group comparisons
        """
        df = pd.DataFrame({
            'prediction': y_pred,
            'group': protected_attribute
        })
        
        # Calculate selection rate (% predicted HIGH RISK or 1) for each group
        selection_rates = df.groupby('group')['prediction'].agg([
            ('count', 'count'),
            ('selected', 'sum'),
            ('selection_rate', 'mean')
        ]).reset_index()
        
        # If reference group not specified, use group with highest selection rate
        if reference_group is None:
            reference_group = selection_rates.loc[
                selection_rates['selection_rate'].idxmax(), 'group'
            ]
        
        reference_rate = selection_rates[
            selection_rates['group'] == reference_group
        ]['selection_rate'].values[0]
        
        # Calculate adverse impact ratios
        selection_rates['adverse_impact_ratio'] = (
            selection_rates['selection_rate'] / reference_rate
        )
        
        # Flag violations
        selection_rates['passes_fairness'] = (
            (selection_rates['adverse_impact_ratio'] >= self.threshold_lower) &
            (selection_rates['adverse_impact_ratio'] <= self.threshold_upper)
        )
        
        return selection_rates
    
    def equal_opportunity_difference(self, y_true, y_pred, protected_attribute):
        """
        Calculate Equal Opportunity Difference
        Measures difference in True Positive Rate across groups
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        protected_attribute : array-like
            Demographic attribute
        
        Returns:
        --------
        pd.DataFrame
            True Positive Rates by group
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': protected_attribute
        })
        
        results = []
        for group in df['group'].unique():
            group_data = df[df['group'] == group]
            
            # Calculate TPR (Recall) for this group
            # TPR = True Positives / (True Positives + False Negatives)
            tp = ((group_data['y_true'] == 1) & (group_data['y_pred'] == 1)).sum()
            fn = ((group_data['y_true'] == 1) & (group_data['y_pred'] == 0)).sum()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            results.append({
                'group': group,
                'true_positives': tp,
                'false_negatives': fn,
                'true_positive_rate': tpr
            })
        
        tpr_df = pd.DataFrame(results)
        
        # Calculate max difference
        tpr_diff = tpr_df['true_positive_rate'].max() - tpr_df['true_positive_rate'].min()
        
        print(f"\nEqual Opportunity Analysis:")
        print(f"Max TPR difference: {tpr_diff:.4f}")
        print(f"{'PASS' if tpr_diff < 0.10 else 'FAIL'}: Difference should be <10 percentage points")
        
        return tpr_df
    
    def predictive_parity(self, y_true, y_pred, protected_attribute):
        """
        Calculate Predictive Parity (Precision equality across groups)
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels  
        protected_attribute : array-like
            Demographic attribute
        
        Returns:
        --------
        pd.DataFrame
            Precision scores by group
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'group': protected_attribute
        })
        
        results = []
        for group in df['group'].unique():
            group_data = df[df['group'] == group]
            
            # Calculate Precision for this group
            # Precision = True Positives / (True Positives + False Positives)
            tp = ((group_data['y_true'] == 1) & (group_data['y_pred'] == 1)).sum()
            fp = ((group_data['y_true'] == 0) & (group_data['y_pred'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            results.append({
                'group': group,
                'true_positives': tp,
                'false_positives': fp,
                'precision': precision
            })
        
        precision_df = pd.DataFrame(results)
        
        # Calculate max difference
        precision_diff = precision_df['precision'].max() - precision_df['precision'].min()
        
        print(f"\nPredictive Parity Analysis:")
        print(f"Max Precision difference: {precision_diff:.4f}")
        print(f"{'PASS' if precision_diff < 0.10 else 'FAIL'}: Difference should be <10 percentage points")
        
        return precision_df
    
    def comprehensive_fairness_audit(self, y_true, y_pred, demographic_data, 
                                    demographic_cols):
        """
        Run complete fairness audit across multiple demographic attributes
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels (binary: 0 or 1)
        demographic_data : pd.DataFrame
            DataFrame with demographic attributes
        demographic_cols : list
            List of column names to test for fairness
        
        Returns:
        --------
        dict
            Complete audit results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE FAIRNESS AUDIT")
        print("="*70)
        
        audit_results = {}
        
        for col in demographic_cols:
            print(f"\n{'─'*70}")
            print(f"Testing: {col}")
            print(f"{'─'*70}")
            
            # Adverse Impact Ratio
            print(f"\n1. Adverse Impact Ratio (EEOC 4/5ths Rule)")
            print(f"   Target range: {self.threshold_lower} - {self.threshold_upper}")
            
            air_results = self.adverse_impact_ratio(
                y_pred, 
                demographic_data[col]
            )
            
            print(air_results.to_string(index=False))
            
            # Check if all groups pass
            all_pass = air_results['passes_fairness'].all()
            violations = air_results[~air_results['passes_fairness']]
            
            if all_pass:
                print(f"\n✓ PASS: All groups within acceptable range")
            else:
                print(f"\n✗ FAIL: {len(violations)} group(s) outside acceptable range")
                print(f"   Violations:")
                for _, row in violations.iterrows():
                    print(f"   - {row['group']}: Ratio = {row['adverse_impact_ratio']:.3f}")
            
            # Equal Opportunity
            print(f"\n2. Equal Opportunity (True Positive Rate Equality)")
            tpr_results = self.equal_opportunity_difference(
                y_true,
                y_pred,
                demographic_data[col]
            )
            print(tpr_results.to_string(index=False))
            
            # Predictive Parity
            print(f"\n3. Predictive Parity (Precision Equality)")
            precision_results = self.predictive_parity(
                y_true,
                y_pred,
                demographic_data[col]
            )
            print(precision_results.to_string(index=False))
            
            # Store results
            audit_results[col] = {
                'adverse_impact': air_results,
                'equal_opportunity': tpr_results,
                'predictive_parity': precision_results,
                'overall_pass': all_pass
            }
        
        # Summary
        print(f"\n" + "="*70)
        print("AUDIT SUMMARY")
        print("="*70)
        
        for col, results in audit_results.items():
            status = "✓ PASS" if results['overall_pass'] else "✗ FAIL"
            print(f"{col:30s} {status}")
        
        all_tests_pass = all(r['overall_pass'] for r in audit_results.values())
        
        print("\n" + "="*70)
        if all_tests_pass:
            print("OVERALL: ✓ MODEL PASSES ALL FAIRNESS TESTS")
        else:
            print("OVERALL: ✗ MODEL REQUIRES BIAS MITIGATION")
        print("="*70 + "\n")
        
        self.audit_results = audit_results
        return audit_results
    
    def generate_fairness_report(self, output_path='reports/fairness_audit.txt'):
        """
        Generate text report of fairness audit results
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("FAIRNESS AUDIT REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Audit Date: {pd.Timestamp.now()}\n")
            f.write(f"Threshold Range: {self.threshold_lower} - {self.threshold_upper}\n\n")
            
            for col, results in self.audit_results.items():
                f.write(f"\n{col.upper()}\n")
                f.write("─"*70 + "\n")
                
                f.write("\nAdverse Impact Ratios:\n")
                f.write(results['adverse_impact'].to_string(index=False))
                f.write(f"\n\nStatus: {'PASS' if results['overall_pass'] else 'FAIL'}\n")
        
        print(f"✓ Fairness report saved to {output_path}")


def create_age_groups(ages):
    """Helper function to bin ages into groups"""
    return pd.cut(
        ages,
        bins=[0, 30, 45, 60, 100],
        labels=['<30', '30-45', '45-60', '>60']
    )


if __name__ == "__main__":
    print("Fairness Testing Module")
    print("Use this module within notebooks or model evaluation scripts")
    print("\nExample usage:")
    print("""
    auditor = FairnessAuditor()
    results = auditor.comprehensive_fairness_audit(
        y_true=y_test,
        y_pred=y_pred,
        demographic_data=test_demographics,
        demographic_cols=['Age_Group', 'Location', 'Employment_Status']
    )
    """)
