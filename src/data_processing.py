"""
Data Processing Pipeline for Credit Risk Model - Task 3

This script transforms raw transaction data into a model-ready dataset by performing:

1. Aggregate Features - Customer-level RFM metrics
2. Extract Time Features - Hour, Day, Month, Year from timestamps
3. Encode Categorical Variables - One-hot and label encoding
4. Handle Missing Values - Median imputation for numerical, mode for categorical
5. Normalize/Standardize Features - StandardScaler for numerical features
6. WOE Transformation - Weight of Evidence for credit scoring (using xverse)

Author: Bati Bank Analytics Team
Date: June 1, 2026
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ============================================
# 1. CREATE AGGREGATE FEATURES
# ============================================

def create_aggregate_features(df):
    """
    Create customer-level aggregate features from transaction data.
    
    Features created:
    - Total Transaction Amount: Sum of all transaction amounts per customer (positive only)
    - Average Transaction Amount: Average transaction amount per customer (positive only)
    - Transaction Count: Number of transactions per customer (all transactions)
    - Standard Deviation of Transaction Amounts: Variability per customer (positive only)
    """
    
    data = df.copy()
    
    # Get all unique customer IDs
    all_customers = data['CustomerId'].unique()
    
    # Filter to only positive amounts (customer spending)
    spending_data = data[data['Amount'] > 0]
    
    # Total Transaction Amount (sum of spending)
    total_amount = spending_data.groupby('CustomerId')['Amount'].sum()
    
    # Average Transaction Amount (mean of spending)
    avg_amount = spending_data.groupby('CustomerId')['Amount'].mean()
    
    # Transaction Count (all transactions)
    transaction_count = data.groupby('CustomerId')['TransactionId'].count()
    
    # Standard Deviation (spending variability)
    std_amount = spending_data.groupby('CustomerId')['Amount'].std()
    
    # Reindex all series to include ALL customers (fill missing with 0)
    total_amount = total_amount.reindex(all_customers, fill_value=0)
    avg_amount = avg_amount.reindex(all_customers, fill_value=0)
    transaction_count = transaction_count.reindex(all_customers, fill_value=0)
    std_amount = std_amount.reindex(all_customers, fill_value=0)
    
    # Combine into DataFrame
    customer_features = pd.DataFrame({
        'CustomerId': all_customers,
        'TotalTransactionAmount': total_amount.values,
        'AverageTransactionAmount': avg_amount.values,
        'TransactionCount': transaction_count.values,
        'StdDevTransactionAmount': std_amount.values
    })
    
    customer_features = customer_features.sort_values('CustomerId').reset_index(drop=True)
    
    print("   ✅ Aggregate features created")
    print(f"      - {len(customer_features)} customers")
    print(f"      - Features: TotalTransactionAmount, AverageTransactionAmount, TransactionCount, StdDevTransactionAmount")
    
    return customer_features


# ============================================
# 2. EXTRACT TIME FEATURES
# ============================================

def extract_time_features(df):
    """
    Extract time-based features from TransactionStartTime.
    
    Features extracted:
    - TransactionHour: Hour of the day (0-23)
    - TransactionDay: Day of the month (1-31)
    - TransactionMonth: Month of the year (1-12)
    - TransactionYear: Year of the transaction
    """
    
    data = df.copy()
    
    if 'TransactionStartTime' not in data.columns:
        raise ValueError("TransactionStartTime column not found in dataset")
    
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    
    print("   ✅ Time features extracted")
    print(f"      - TransactionHour: {data['TransactionHour'].min()} to {data['TransactionHour'].max()}")
    print(f"      - TransactionDay: {data['TransactionDay'].min()} to {data['TransactionDay'].max()}")
    print(f"      - TransactionMonth: {data['TransactionMonth'].min()} to {data['TransactionMonth'].max()}")
    print(f"      - TransactionYear: {data['TransactionYear'].min()} to {data['TransactionYear'].max()}")
    
    return data


# ============================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================

def encode_categorical_features(df):
    """
    Encode categorical variables into numerical format.
    
    Methods used:
    - One-Hot Encoding: For ProductCategory (creates binary columns)
    - Label Encoding: For ChannelId (assigns unique integer to each channel)
    """
    
    data = df.copy()
    
    # One-Hot Encoding for ProductCategory
    if 'ProductCategory' in data.columns:
        unique_categories = data['ProductCategory'].nunique()
        print(f"      - ProductCategory: {unique_categories} unique values")
        
        one_hot_encoded = pd.get_dummies(data['ProductCategory'], prefix='category')
        data = pd.concat([data, one_hot_encoded], axis=1)
        data = data.drop(columns=['ProductCategory'])
        
        print(f"      - One-Hot Encoding: created {one_hot_encoded.shape[1]} binary columns")
    
    # Label Encoding for ChannelId
    if 'ChannelId' in data.columns:
        unique_channels = sorted(data['ChannelId'].unique())
        channel_mapping = {channel: idx for idx, channel in enumerate(unique_channels)}
        
        data['ChannelId_Encoded'] = data['ChannelId'].map(channel_mapping)
        data = data.drop(columns=['ChannelId'])
        
        print(f"      - Label Encoding: ChannelId mapped to values 0-{len(unique_channels)-1}")
    
    return data


# ============================================
# 4. HANDLE MISSING VALUES
# ============================================

def handle_missing_values(df):
    """
    Handle missing values using imputation strategies.
    
    Methods:
    - Numerical columns: Median imputation (robust to outliers)
    - Categorical columns: Mode imputation (most frequent value)
    """
    
    data = df.copy()
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if 'CustomerId' in numerical_cols:
        numerical_cols.remove('CustomerId')
    
    missing_before = data.isnull().sum().sum()
    
    if missing_before == 0:
        print("   ✅ No missing values found - no imputation needed")
        return data
    
    print(f"   ⚠️ Found {missing_before} missing values before imputation")
    
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
    
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_value = data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown'
            data[col] = data[col].fillna(mode_value)
    
    missing_after = data.isnull().sum().sum()
    print(f"   ✅ Missing values reduced: {missing_before} → {missing_after}")
    
    return data


# ============================================
# 5. NORMALIZE/STANDARDIZE NUMERICAL FEATURES
# ============================================

def scale_numerical_features(df, method='standardize'):
    """
    Scale numerical features to bring them onto a similar scale.
    """
    
    data = df.copy()
    
    binary_cols = [col for col in data.columns if col.startswith(('category_', 'ChannelId_Encoded'))]
    exclude_cols = ['CustomerId'] + binary_cols
    scale_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude_cols]
    
    if method == 'standardize':
        scaler = StandardScaler()
        data[scale_cols] = scaler.fit_transform(data[scale_cols])
        print(f"   ✅ Standardization applied (mean=0, std=1)")
        print(f"      - Scaled columns: {len(scale_cols)} features")
        
    elif method == 'normalize':
        scaler = MinMaxScaler()
        data[scale_cols] = scaler.fit_transform(data[scale_cols])
        print(f"   ✅ Normalization applied (range [0, 1])")
        print(f"      - Scaled columns: {len(scale_cols)} features")
    
    return data


# ============================================
# 6. WOE TRANSFORMATION (using xverse)
# ============================================

def apply_woe_transformation(df, target_column='Target_Binary', feature_columns=None):
    """
    Apply Weight of Evidence (WOE) transformation to categorical features.
    """
    
    try:
        from xverse.transformer import WOE
    except ImportError:
        print("   ⚠️ xverse not installed. Run: pip install xverse")
        return df, pd.DataFrame()
    
    data = df.copy()
    
    if target_column not in data.columns:
        print(f"   ⚠️ Target column '{target_column}' not found. Skipping WOE.")
        return data, pd.DataFrame()
    
    # Select categorical columns
    if feature_columns is None:
        feature_columns = data.select_dtypes(include=['object']).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in [target_column, 'CustomerId']]
    
    if len(feature_columns) == 0:
        print("   ⚠️ No categorical features found for WOE transformation.")
        return data, pd.DataFrame()
    
    print(f"   📊 Applying WOE transformation to {len(feature_columns)} features:")
    for col in feature_columns:
        print(f"      - {col}")
    
    # Fill missing values
    for col in feature_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna('missing')
    
    X = data[feature_columns].copy()
    y = data[target_column].copy()
    
    woe_transformer = WOE()
    woe_transformer.fit(X, y)
    
    X_woe = woe_transformer.transform(X)
    iv_df = woe_transformer.iv_df.copy()
    
    def interpret_iv(value):
        if value < 0.02:
            return "Not useful"
        elif value < 0.1:
            return "Weak"
        elif value < 0.3:
            return "Medium"
        elif value < 0.5:
            return "Strong"
        else:
            return "Suspicious"
    
    iv_df['Predictiveness'] = iv_df['Information_Value'].apply(interpret_iv)
    
    print(f"\n   ✅ WOE transformation complete")
    print(f"   📊 Information Value (IV) for feature selection:")
    
    for _, row in iv_df.iterrows():
        iv_value = row['Information_Value']
        var_name = row['Variable_Name']
        strength = row['Predictiveness']
        
        if iv_value >= 0.1:
            indicator = "✅ KEEP"
        elif iv_value >= 0.02:
            indicator = "⚠️  MAYBE"
        else:
            indicator = "❌ DROP"
        
        print(f"      {indicator} {var_name}: {iv_value:.4f} ({strength})")
    
    # Drop original categorical columns and add WOE columns
    for col in feature_columns:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    for col in X_woe.columns:
        data[col] = X_woe[col]
    
    return data, iv_df


# ============================================
# PROXY TARGET CREATION
# ============================================

def create_proxy_target(customer_df):
    """
    Create proxy target variable using RFM scoring.
    """
    
    data = customer_df.copy()
    
    print("\n📌 Creating Proxy Target Variable (RFM-based)")
    print("-" * 40)
    
    data['R_rank'] = data['TransactionCount'].rank(ascending=False)
    data['F_rank'] = data['TransactionCount'].rank(ascending=True)
    data['M_rank'] = data['TotalTransactionAmount'].rank(ascending=True)
    
    data['R_score'] = data['R_rank'] / data['R_rank'].max()
    data['F_score'] = data['F_rank'] / data['F_rank'].max()
    data['M_score'] = data['M_rank'] / data['M_rank'].max()
    
    data['RFM_Score'] = (data['R_score'] + data['F_score'] + data['M_score']) / 3
    
    high_threshold = data['RFM_Score'].quantile(0.80)
    low_threshold = data['RFM_Score'].quantile(0.20)
    
    data['Target'] = 'middle'
    data.loc[data['RFM_Score'] >= high_threshold, 'Target'] = 'good'
    data.loc[data['RFM_Score'] <= low_threshold, 'Target'] = 'bad'
    
    data['Target_Binary'] = (data['Target'] == 'bad').astype(int)
    
    good_count = (data['Target'] == 'good').sum()
    bad_count = (data['Target'] == 'bad').sum()
    
    print(f"   ✅ Proxy target created")
    print(f"      - Good customers (low risk, Target=0): {good_count}")
    print(f"      - Bad customers (high risk, Target=1): {bad_count}")
    print(f"      - Middle segment (excluded): {len(data) - good_count - bad_count}")
    
    return data


# ============================================
# MAIN PIPELINE
# ============================================

def run_data_pipeline(raw_data_path='data/raw/data.csv', 
                      processed_data_path='data/processed/',
                      scaling_method='standardize',
                      apply_woe=True):
    """
    Run the complete data processing pipeline.
    """
    
    print("=" * 70)
    print("DATA PROCESSING PIPELINE - TASK 3")
    print("=" * 70)
    
    # Load raw data
    print("\n📂 STEP 0: Loading Raw Data")
    print("-" * 40)
    df = pd.read_csv(raw_data_path)
    print(f"   Raw data shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # STEP 1: Extract Time Features
    print("\n📌 STEP 1: Extract Time Features")
    print("-" * 40)
    df = extract_time_features(df)
    
    # STEP 2: Encode Categorical Variables
    print("\n📌 STEP 2: Encode Categorical Variables")
    print("-" * 40)
    df = encode_categorical_features(df)
    
    # STEP 3: Handle Missing Values
    print("\n📌 STEP 3: Handle Missing Values")
    print("-" * 40)
    df = handle_missing_values(df)
    
    # STEP 4: Create Aggregate Features
    print("\n📌 STEP 4: Create Aggregate Features")
    print("-" * 40)
    customer_features = create_aggregate_features(df)
    
    # STEP 5: Scale Numerical Features
    print("\n📌 STEP 5: Scale Numerical Features")
    print("-" * 40)
    customer_features_scaled = scale_numerical_features(customer_features, method=scaling_method)
    
    # STEP 6: Create Proxy Target
    print("\n📌 STEP 6: Create Proxy Target")
    print("-" * 40)
    customer_features_with_target = create_proxy_target(customer_features_scaled)
    
    # STEP 7: Apply WOE Transformation (optional)
    iv_df = None
    if apply_woe:
        print("\n📌 STEP 7: Apply WOE Transformation")
        print("-" * 40)
        customer_features_with_target, iv_df = apply_woe_transformation(
            customer_features_with_target, 
            target_column='Target_Binary'
        )
    
    # Save outputs
    print("\n📁 Saving Processed Data")
    print("-" * 40)
    os.makedirs(processed_data_path, exist_ok=True)
    
    transaction_output = os.path.join(processed_data_path, 'transactions_processed.csv')
    df.to_csv(transaction_output, index=False)
    print(f"   ✅ Saved: {transaction_output}")
    
    customer_output = os.path.join(processed_data_path, 'customer_features_model_ready.csv')
    customer_features_with_target.to_csv(customer_output, index=False)
    print(f"   ✅ Saved: {customer_output}")
    
    if iv_df is not None and len(iv_df) > 0:
        iv_output = os.path.join(processed_data_path, 'information_values.csv')
        iv_df.to_csv(iv_output, index=False)
        print(f"   ✅ Saved: {iv_output}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Model-Ready Dataset:")
    print(f"   - Shape: {customer_features_with_target.shape[0]} customers, {customer_features_with_target.shape[1]} features")
    print(f"   - WOE applied: {apply_woe}")
    print("=" * 70)
    
    return customer_features_with_target, df, iv_df


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    model_ready_data, transactions_data, iv_scores = run_data_pipeline(
        raw_data_path='data/raw/data.csv',
        processed_data_path='data/processed/',
        scaling_method='standardize',
        apply_woe=False
    
    )
if __name__ == "__main__":
   model_ready_data, transactions_data, iv_scores = run_data_pipeline(
    raw_data_path='data/raw/data.csv',
        processed_data_path='data/processed/',
        scaling_method='standardize',
        apply_woe=False  # Change this from True to False
    )