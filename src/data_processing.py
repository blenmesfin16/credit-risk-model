"""
Data Processing Pipeline for Credit Risk Model - Tasks 3 & 4

Task 3 - Feature Engineering:
1. Aggregate Features - Customer-level RFM metrics
2. Extract Time Features - Hour, Day, Month, Year from timestamps
3. Encode Categorical Variables - One-hot and label encoding
4. Handle Missing Values - Median imputation for numerical, mode for categorical
5. Normalize/Standardize Features - StandardScaler for numerical features

Task 4 - Proxy Target Variable Engineering:
6. K-Means Clustering on RFM metrics to identify high-risk customers

Author: Bati Bank Analytics Team
Date: June 2, 2026
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


# ============================================
# TASK 3: FEATURE ENGINEERING FUNCTIONS
# ============================================

def create_aggregate_features(df):
    """
    Create customer-level aggregate features from transaction data.
    
    Features created:
    - Total Transaction Amount: Sum of all transaction amounts per customer (positive only)
    - Average Transaction Amount: Average transaction amount per customer (positive only)
    - Transaction Count: Number of transactions per customer (all transactions)
    - Standard Deviation of Transaction Amounts: Variability per customer (positive only)
    - Recency: Days since last transaction
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
    
    # Calculate Recency (days since last transaction)
    if 'TransactionStartTime' in data.columns:
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        last_transaction = data.groupby('CustomerId')['TransactionStartTime'].max()
        reference_date = data['TransactionStartTime'].max()
        recency = (reference_date - last_transaction).dt.days
        recency = recency.reindex(all_customers, fill_value=365)
        customer_features['Recency'] = recency.values
    else:
        customer_features['Recency'] = customer_features['TransactionCount'].rank(ascending=False)
    
    customer_features = customer_features.sort_values('CustomerId').reset_index(drop=True)
    
    print("   ✅ Aggregate features created")
    print(f"      - {len(customer_features)} customers")
    print(f"      - Features: TotalAmount, AvgAmount, Count, StdDev, Recency")
    
    return customer_features


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


def scale_numerical_features(df, method='standardize'):
    """
    Scale numerical features to bring them onto a similar scale.
    
    Parameters:
    - method: 'standardize' (mean=0, std=1) or 'normalize' (range [0,1])
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
# TASK 4: PROXY TARGET VARIABLE ENGINEERING
# Using K-Means Clustering on RFM Metrics
# ============================================

def create_proxy_target_kmeans(customer_df, random_state=42):
    """
    Create proxy target variable using K-Means clustering on RFM metrics.
    
    This method segments customers into 3 groups based on their RFM profile
    and identifies the high-risk cluster (least engaged customers).
    
    Parameters:
    - customer_df: DataFrame with RFM features (TransactionCount, TotalTransactionAmount, Recency)
    - random_state: Seed for reproducibility
    
    Returns:
    - DataFrame with added 'is_high_risk' column
    - Cluster analysis summary
    """
    
    data = customer_df.copy()
    
    print("\n" + "=" * 70)
    print("TASK 4: PROXY TARGET VARIABLE ENGINEERING (K-MEANS CLUSTERING)")
    print("=" * 70)
    
    # Step 1: Prepare RFM features
    print("\n📌 STEP 1: Preparing RFM Features")
    print("-" * 40)
    
    rfm_features = ['Recency', 'TransactionCount', 'TotalTransactionAmount']
    available_features = [f for f in rfm_features if f in data.columns]
    
    print(f"   RFM features for clustering: {available_features}")
    
    # Step 2: Scale features
    print("\n📌 STEP 2: Scaling RFM Features")
    print("-" * 40)
    
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(data[available_features])
    print(f"   ✅ Features scaled (mean=0, std=1)")
    
    # Step 3: K-Means clustering
    print("\n📌 STEP 3: K-Means Clustering (3 segments)")
    print("-" * 40)
    
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_rfm)
    
    print(f"   ✅ K-Means clustering complete")
    print(f"   Cluster distribution:")
    for cluster in sorted(data['cluster'].unique()):
        count = (data['cluster'] == cluster).sum()
        pct = count / len(data) * 100
        print(f"      Cluster {cluster}: {count} customers ({pct:.1f}%)")
    
    # Step 4: Analyze clusters
    print("\n📌 STEP 4: Analyzing Cluster Profiles")
    print("-" * 40)
    
    cluster_profiles = data.groupby('cluster')[available_features].mean()
    
    print("\n   Cluster Profiles (mean values):")
    for cluster in cluster_profiles.index:
        values = []
        for col in available_features:
            val = cluster_profiles.loc[cluster, col]
            if col == 'Recency':
                values.append(f"{col}={val:.1f} days")
            elif col == 'TransactionCount':
                values.append(f"{col}={val:.1f} txns")
            else:
                values.append(f"{col}={val:.0f}")
        print(f"      Cluster {cluster}: {', '.join(values)}")
    
    # Identify high-risk cluster (lowest engagement)
    cluster_scores = {}
    for cluster in cluster_profiles.index:
        score = 0
        for col in available_features:
            if col == 'Recency':
                rank = cluster_profiles[col].rank(ascending=False).loc[cluster]
            else:
                rank = cluster_profiles[col].rank(ascending=True).loc[cluster]
            score += rank
        cluster_scores[cluster] = score
    
    high_risk_cluster = min(cluster_scores, key=cluster_scores.get)
    
    print(f"\n   🎯 High-Risk Cluster Identified: Cluster {high_risk_cluster}")
    print(f"      (Least engaged - high recency, low frequency, low monetary)")
    
    # Step 5: Create binary target
    print("\n📌 STEP 5: Creating 'is_high_risk' Binary Target")
    print("-" * 40)
    
    data['is_high_risk'] = (data['cluster'] == high_risk_cluster).astype(int)
    
    high_risk_count = data['is_high_risk'].sum()
    low_risk_count = len(data) - high_risk_count
    
    print(f"   ✅ Binary target created")
    print(f"      - High Risk (is_high_risk = 1): {high_risk_count} ({high_risk_count/len(data)*100:.1f}%)")
    print(f"      - Low Risk (is_high_risk = 0): {low_risk_count} ({low_risk_count/len(data)*100:.1f}%)")
    
    # Step 6: Summary statistics
    print("\n📌 STEP 6: Cluster Summary Statistics")
    print("-" * 40)
    
    summary = data.groupby('cluster').agg({
        'TransactionCount': ['count', 'mean'],
        'TotalTransactionAmount': ['mean'],
        'is_high_risk': 'first'
    }).round(2)
    print(summary)
    
    # Drop temporary cluster column
    data = data.drop(columns=['cluster'])
    
    return data, cluster_profiles, high_risk_cluster


# ============================================
# MAIN PIPELINE - TASKS 3 & 4
# ============================================

def run_data_pipeline(raw_data_path='data/raw/data.csv', 
                      processed_data_path='data/processed/',
                      scaling_method='standardize'):
    """
    Run the complete data processing pipeline (Tasks 3 & 4).
    
    Steps:
    Task 3:
    1. Extract time features from timestamp
    2. Encode categorical variables
    3. Handle missing values
    4. Create aggregate features (customer-level)
    5. Scale numerical features
    
    Task 4:
    6. K-Means clustering on RFM to create is_high_risk target
    """
    
    print("=" * 70)
    print("DATA PROCESSING PIPELINE - TASKS 3 & 4")
    print("=" * 70)
    
    # Load raw data
    print("\n📂 STEP 0: Loading Raw Data")
    print("-" * 40)
    df = pd.read_csv(raw_data_path)
    print(f"   Raw data shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # ============================================
    # TASK 3: FEATURE ENGINEERING
    # ============================================
    print("\n" + "=" * 70)
    print("TASK 3: FEATURE ENGINEERING")
    print("=" * 70)
    
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
    
    # STEP 4: Create Aggregate Features (customer-level)
    print("\n📌 STEP 4: Create Aggregate Features")
    print("-" * 40)
    customer_features = create_aggregate_features(df)
    
    # STEP 5: Scale Numerical Features
    print("\n📌 STEP 5: Scale Numerical Features")
    print("-" * 40)
    customer_features_scaled = scale_numerical_features(customer_features, method=scaling_method)
    
    # ============================================
    # TASK 4: PROXY TARGET ENGINEERING
    # ============================================
    print("\n" + "=" * 70)
    print("TASK 4: PROXY TARGET ENGINEERING")
    print("=" * 70)
    
    customer_features_with_target, cluster_profiles, high_risk_cluster = create_proxy_target_kmeans(
        customer_features_scaled,
        random_state=42
    )
    
    # ============================================
    # SAVE OUTPUTS
    # ============================================
    print("\n📁 Saving Processed Data")
    print("-" * 40)
    os.makedirs(processed_data_path, exist_ok=True)
    
    # Save transaction-level data
    transaction_output = os.path.join(processed_data_path, 'transactions_processed.csv')
    df.to_csv(transaction_output, index=False)
    print(f"   ✅ Saved: {transaction_output}")
    
    # Save customer-level features with target
    customer_output = os.path.join(processed_data_path, 'customer_features_model_ready.csv')
    customer_features_with_target.to_csv(customer_output, index=False)
    print(f"   ✅ Saved: {customer_output}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Final Model-Ready Dataset:")
    print(f"   - Shape: {customer_features_with_target.shape[0]} customers, {customer_features_with_target.shape[1]} features")
    print(f"   - Target column: is_high_risk")
    print(f"   - High risk customers (1): {customer_features_with_target['is_high_risk'].sum()}")
    print(f"   - Low risk customers (0): {len(customer_features_with_target) - customer_features_with_target['is_high_risk'].sum()}")
    print("=" * 70)
    
    return customer_features_with_target, df


# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    model_ready_data, transactions_data = run_data_pipeline(
        raw_data_path='data/raw/data.csv',
        processed_data_path='data/processed/',
        scaling_method='standardize'
    )
    print("\n🎯 Data processing complete! Ready for model training.")