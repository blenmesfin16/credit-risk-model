"""
Model Training Script for Credit Risk Prediction - Task 5
Complete Model Evaluation with All Metrics

Author: Bati Bank Analytics Team
Date: June 2, 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import warnings
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

print("=" * 70)
print("TASK 5: MODEL TRAINING WITH COMPLETE EVALUATION")
print("=" * 70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================

print("\n📂 STEP 1: Loading Processed Data")
print("-" * 40)

df = pd.read_csv('data/processed/customer_features_model_ready.csv')
print(f"   Data shape: {df.shape}")

# Handle missing values
numerical_cols = ['TotalTransactionAmount', 'AverageTransactionAmount', 
                  'TransactionCount', 'StdDevTransactionAmount', 'Recency']

for col in numerical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Prepare features and target
exclude_cols = ['CustomerId', 'is_high_risk']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols].fillna(0)
y = df['is_high_risk']

print(f"   Features: {len(feature_cols)}")
print(f"   Target distribution - High risk (1): {y.sum()}, Low risk (0): {len(y)-y.sum()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n   Train set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# ============================================
# STEP 2: DEFINE EVALUATION FUNCTION
# ============================================

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Comprehensive model evaluation with all metrics.
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Additional metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Derived metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'mcc': mcc,
        'kappa': kappa,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'positive_predictive_value': positive_predictive_value,
        'negative_predictive_value': negative_predictive_value
    }
    
    return metrics, cm


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"      📊 Confusion matrix saved: {save_path}")
    plt.close()


def print_evaluation_report(metrics, model_name):
    """
    Print formatted evaluation report.
    """
    print(f"\n   📈 EVALUATION REPORT: {model_name}")
    print("   " + "-" * 40)
    print(f"   🎯 Primary Metrics:")
    print(f"      - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      - Precision: {metrics['precision']:.4f}")
    print(f"      - Recall:    {metrics['recall']:.4f}")
    print(f"      - F1 Score:  {metrics['f1_score']:.4f}")
    print(f"      - ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\n   📊 Additional Metrics:")
    print(f"      - Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"      - Specificity:       {metrics['specificity']:.4f}")
    print(f"      - MCC:               {metrics['mcc']:.4f}")
    print(f"      - Cohen's Kappa:     {metrics['kappa']:.4f}")
    print(f"\n   ⚠️ Error Rates:")
    print(f"      - False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"      - False Negative Rate: {metrics['false_negative_rate']:.4f}")

# ============================================
# STEP 3: DEFINE MODELS WITH HYPERPARAMETER GRIDS
# ============================================

print("\n📌 STEP 2: Defining Hyperparameter Grids")
print("-" * 40)

# Logistic Regression
logistic_params = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Decision Tree
dt_params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

print("   Hyperparameter grids defined")

# ============================================
# STEP 4: CREATE EXPERIMENT
# ============================================

experiment_name = "Credit_Risk_Models_Complete_Evaluation"
mlflow.set_experiment(experiment_name)

print(f"\n📌 STEP 3: MLflow Experiment Created")
print(f"   Experiment Name: {experiment_name}")

# ============================================
# STEP 5: TRAIN AND EVALUATE EACH MODEL
# ============================================

print("\n" + "=" * 70)
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("=" * 70)

# Store results
all_results = []
best_auc = 0
best_model_info = None

# Create directory for confusion matrices
os.makedirs('models/confusion_matrices', exist_ok=True)

# ============================================
# 5.1 LOGISTIC REGRESSION
# ============================================

print("\n" + "=" * 50)
print("📊 Model 1: Logistic Regression")
print("=" * 50)

logistic_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

logistic_grid = GridSearchCV(
    logistic_pipeline, logistic_params, cv=5, scoring='roc_auc', n_jobs=-1
)

with mlflow.start_run(run_name="Logistic_Regression") as run:
    logistic_grid.fit(X_train, y_train)
    best_lr = logistic_grid.best_estimator_
    
    # Predictions
    y_pred = best_lr.predict(X_test)
    y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
    
    # Comprehensive evaluation
    metrics, cm = evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    # Plot confusion matrix
    cm_path = f'models/confusion_matrices/logistic_regression_cm.png'
    plot_confusion_matrix(cm, "Logistic Regression", cm_path)
    
    # Log all metrics to MLflow
    mlflow.log_params(logistic_grid.best_params_)
    mlflow.log_param("model_type", "Logistic Regression")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log confusion matrix as artifact
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
    
    # Log model
    mlflow.sklearn.log_model(best_lr, "model")
    
    # Print report
    print_evaluation_report(metrics, "Logistic Regression")
    
    all_results.append({
        'model': 'Logistic Regression',
        'run_id': run.info.run_id,
        **metrics
    })
    
    if metrics['roc_auc'] > best_auc:
        best_auc = metrics['roc_auc']
        best_model_info = ('Logistic Regression', best_lr, run.info.run_id, metrics)

# ============================================
# 5.2 DECISION TREE
# ============================================

print("\n" + "=" * 50)
print("📊 Model 2: Decision Tree")
print("=" * 50)

dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_grid = GridSearchCV(dt_model, dt_params, cv=5, scoring='roc_auc', n_jobs=-1)

with mlflow.start_run(run_name="Decision_Tree") as run:
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    
    y_pred = best_dt.predict(X_test)
    y_pred_proba = best_dt.predict_proba(X_test)[:, 1]
    
    metrics, cm = evaluate_model(y_test, y_pred, y_pred_proba, "Decision Tree")
    
    cm_path = f'models/confusion_matrices/decision_tree_cm.png'
    plot_confusion_matrix(cm, "Decision Tree", cm_path)
    
    mlflow.log_params(dt_grid.best_params_)
    mlflow.log_param("model_type", "Decision Tree")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
    mlflow.sklearn.log_model(best_dt, "model")
    
    print_evaluation_report(metrics, "Decision Tree")
    
    all_results.append({
        'model': 'Decision Tree',
        'run_id': run.info.run_id,
        **metrics
    })
    
    if metrics['roc_auc'] > best_auc:
        best_auc = metrics['roc_auc']
        best_model_info = ('Decision Tree', best_dt, run.info.run_id, metrics)

# ============================================
# 5.3 RANDOM FOREST
# ============================================

print("\n" + "=" * 50)
print("📊 Model 3: Random Forest")
print("=" * 50)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)

with mlflow.start_run(run_name="Random_Forest") as run:
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    metrics, cm = evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    
    cm_path = f'models/confusion_matrices/random_forest_cm.png'
    plot_confusion_matrix(cm, "Random Forest", cm_path)
    
    mlflow.log_params(rf_grid.best_params_)
    mlflow.log_param("model_type", "Random Forest")
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log feature importances
    for col, imp in zip(feature_cols, best_rf.feature_importances_):
        mlflow.log_metric(f"importance_{col}", imp)
    
    mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
    mlflow.sklearn.log_model(best_rf, "model")
    
    print_evaluation_report(metrics, "Random Forest")
    
    all_results.append({
        'model': 'Random Forest',
        'run_id': run.info.run_id,
        **metrics
    })
    
    if metrics['roc_auc'] > best_auc:
        best_auc = metrics['roc_auc']
        best_model_info = ('Random Forest', best_rf, run.info.run_id, metrics)

# ============================================
# 5.4 XGBOOST
# ============================================

try:
    from xgboost import XGBClassifier
    
    print("\n" + "=" * 50)
    print("📊 Model 4: XGBoost")
    print("=" * 50)
    
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
    
    with mlflow.start_run(run_name="XGBoost") as run:
        xgb_grid.fit(X_train, y_train)
        best_xgb = xgb_grid.best_estimator_
        
        y_pred = best_xgb.predict(X_test)
        y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
        
        metrics, cm = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        cm_path = f'models/confusion_matrices/xgboost_cm.png'
        plot_confusion_matrix(cm, "XGBoost", cm_path)
        
        mlflow.log_params(xgb_grid.best_params_)
        mlflow.log_param("model_type", "XGBoost")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
        mlflow.sklearn.log_model(best_xgb, "model")
        
        print_evaluation_report(metrics, "XGBoost")
        
        all_results.append({
            'model': 'XGBoost',
            'run_id': run.info.run_id,
            **metrics
        })
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model_info = ('XGBoost', best_xgb, run.info.run_id, metrics)
            
except ImportError:
    print("\n   ⚠️ XGBoost not installed - skipping")

# ============================================
# STEP 6: COMPREHENSIVE MODEL COMPARISON
# ============================================

print("\n" + "=" * 70)
print("STEP 5: COMPREHENSIVE MODEL COMPARISON")
print("=" * 70)

# Create comparison DataFrame
comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

print("\n📊 ALL MODELS - COMPLETE METRICS COMPARISON:")
print("-" * 90)
print(comparison_df[['model', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc', 'mcc']].to_string(index=False))

print("\n" + "=" * 70)
print("STEP 6: BEST MODEL SELECTION")
print("=" * 70)

print(f"\n🏆 BEST MODEL: {best_model_info[0]}")
print(f"   📈 Performance Summary:")
print(f"      - ROC-AUC:      {best_model_info[3]['roc_auc']:.4f}")
print(f"      - Accuracy:     {best_model_info[3]['accuracy']:.4f}")
print(f"      - Precision:    {best_model_info[3]['precision']:.4f}")
print(f"      - Recall:       {best_model_info[3]['recall']:.4f}")
print(f"      - F1 Score:     {best_model_info[3]['f1_score']:.4f}")
print(f"      - Specificity:  {best_model_info[3]['specificity']:.4f}")
print(f"      - MCC:          {best_model_info[3]['mcc']:.4f}")
print(f"      - Kappa:        {best_model_info[3]['kappa']:.4f}")

# ============================================
# STEP 7: REGISTER BEST MODEL
# ============================================

print("\n" + "=" * 70)
print("STEP 7: MLFLOW MODEL REGISTRY")
print("=" * 70)

model_name, best_model, run_id, best_metrics = best_model_info

# Register the model
client = MlflowClient()
model_uri = f"runs:/{run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="Credit_Risk_Best_Model"
)

print(f"\n   ✅ Model registered in MLflow Registry:")
print(f"      - Model Name: Credit_Risk_Best_Model")
print(f"      - Version: {registered_model.version}")

# Transition to Production (since it's the best)
client.transition_model_version_stage(
    name="Credit_Risk_Best_Model",
    version=registered_model.version,
    stage="Production"
)
print(f"   ✅ Model moved to: Production")

# Save locally
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model_production.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print(f"   ✅ Model saved locally: models/best_model_production.pkl")

# Save comparison results
comparison_df.to_csv('models/model_evaluation_complete.csv', index=False)
print(f"   ✅ Complete evaluation saved: models/model_evaluation_complete.csv")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ TASK 5 COMPLETE - COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)
print(f"\n📊 Summary:")
print(f"   - Models evaluated: {len(all_results)}")
print(f"   - Best model: {best_model_info[0]}")
print(f"   - Best ROC-AUC: {best_auc:.4f}")
print(f"\n📁 Saved Files:")
print(f"   - models/best_model_production.pkl")
print(f"   - models/feature_columns.pkl")
print(f"   - models/model_evaluation_complete.csv")
print(f"   - models/confusion_matrices/*.png")
print(f"\n📁 To view MLflow UI, run:")
print(f"   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("=" * 70)