"""
Synthetic Data Generator for Geldium Credit Risk Project
Generates realistic credit card customer data for demonstration purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic credit card customer data
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate (default: 500)
    
    Returns:
    --------
    pd.DataFrame
        Synthetic customer dataset
    """
    
    print(f"Generating {n_samples} synthetic customer records...")
    
    # Generate Customer IDs
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_samples + 1)]
    
    # Generate Demographics
    ages = np.random.normal(45, 15, n_samples).clip(22, 85).astype(int)
    
    # Income (log-normal distribution, some missing)
    incomes = np.random.lognormal(10.8, 0.6, n_samples).clip(20000, 500000)
    income_missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    incomes[income_missing_idx] = np.nan
    
    # Credit Scores (normal distribution, very few missing)
    credit_scores = np.random.normal(680, 80, n_samples).clip(300, 850).astype(int)
    score_missing_idx = np.random.choice(n_samples, size=2, replace=False)
    credit_scores[score_missing_idx] = np.nan
    
    # Credit Utilization (beta distribution skewed right)
    credit_utilization = (np.random.beta(2, 5, n_samples) * 120).clip(0, 150)
    
    # Missed Payments (Poisson distribution)
    missed_payments = np.random.poisson(0.8, n_samples).clip(0, 12)
    
    # Loan Balance (log-normal, some missing)
    loan_balances = np.random.lognormal(9, 1.2, n_samples).clip(0, 100000)
    loan_missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    loan_balances[loan_missing_idx] = np.nan
    
    # Debt-to-Income Ratio (calculated from loan balance and income where possible)
    dti_ratios = np.where(
        np.isnan(incomes) | np.isnan(loan_balances),
        np.nan,
        (loan_balances / (incomes / 12)) * 100
    ).clip(0, 150)
    
    # Employment Status
    employment_statuses = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed'],
        size=n_samples,
        p=[0.75, 0.15, 0.10]
    )
    
    # Account Tenure (years)
    account_tenures = np.random.exponential(3.5, n_samples).clip(0.1, 20)
    
    # Credit Card Type (correlated with credit score)
    credit_card_types = []
    for score in credit_scores:
        if np.isnan(score):
            card_type = 'Standard'
        elif score > 750:
            card_type = np.random.choice(['Platinum', 'Gold'], p=[0.6, 0.4])
        elif score > 650:
            card_type = np.random.choice(['Gold', 'Standard'], p=[0.5, 0.5])
        else:
            card_type = 'Standard'
        credit_card_types.append(card_type)
    
    # Location (fictional cities)
    locations = np.random.choice(
        ['Metro City', 'Riverside', 'Hilltown', 'Coastal Bay', 'Central Valley'],
        size=n_samples,
        p=[0.35, 0.20, 0.20, 0.15, 0.10]
    )
    
    # Payment History (Month_1 to Month_6)
    # Month_1 is most recent: 0=on-time, 1=late, 2=missed
    payment_histories = []
    for i in range(n_samples):
        # Base probability of late/missed payment
        base_prob = 0.15 if missed_payments[i] > 2 else 0.05
        
        history = []
        for month in range(6):
            rand = np.random.random()
            if rand < base_prob:
                payment = 2  # Missed
            elif rand < base_prob * 2:
                payment = 1  # Late
            else:
                payment = 0  # On-time
            history.append(payment)
        
        payment_histories.append(history)
    
    # Target Variable: Delinquent_Account
    # Create realistic delinquency based on risk factors
    delinquency_scores = (
        (missed_payments * 10) +
        (credit_utilization * 0.3) +
        (dti_ratios * 0.2) +
        ((850 - credit_scores) * 0.1) +
        np.random.normal(0, 10, n_samples)
    )
    
    # Convert to binary (top ~12% are delinquent)
    threshold = np.percentile(delinquency_scores[~np.isnan(delinquency_scores)], 88)
    delinquent_accounts = (delinquency_scores > threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Income': incomes,
        'Credit_Score': credit_scores,
        'Credit_Utilization': credit_utilization,
        'Missed_Payments': missed_payments,
        'Delinquent_Account': delinquent_accounts,
        'Loan_Balance': loan_balances,
        'Debt_to_Income_Ratio': dti_ratios,
        'Employment_Status': employment_statuses,
        'Account_Tenure': account_tenures,
        'Credit_Card_Type': credit_card_types,
        'Location': locations,
    })
    
    # Add payment history columns
    for i in range(6):
        df[f'Month_{i+1}'] = [history[i] for history in payment_histories]
    
    print("✓ Synthetic data generated successfully!")
    print(f"  - Total records: {len(df)}")
    print(f"  - Delinquency rate: {df['Delinquent_Account'].mean()*100:.1f}%")
    print(f"  - Missing Income: {df['Income'].isna().sum()} ({df['Income'].isna().mean()*100:.1f}%)")
    print(f"  - Missing Credit Score: {df['Credit_Score'].isna().sum()} ({df['Credit_Score'].isna().mean()*100:.1f}%)")
    
    return df


def save_data(df, output_dir='data/raw'):
    """Save synthetic data to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'delinquency_dataset.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to {filepath}")


if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=500)
    
    # Save to CSV
    save_data(df)
    
    # Display sample
    print("\nSample records:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
