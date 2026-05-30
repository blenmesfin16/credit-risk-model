## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?

The Basel II Accord fundamentally requires that banks can explain how their risk estimates translate into capital requirements. As summarized in the **World Bank Guidelines** (2019, Box 1.2), Basel II is built on three pillars:

- **Pillar 1 (Minimum Capital Requirements):** The model must produce accurate Probability of Default (PD) estimates.
- **Pillar 2 (Supervisory Review):** Regulators must be able to validate the model logic. An opaque "black-box" model cannot pass this review.
- **Pillar 3 (Market Discipline):** Documentation must be transparent enough for external audit.

**Why interpretability is mandatory:**

- Banks using the **Internal Ratings-Based (IRB) approach** must demonstrate that every input feature’s contribution to the risk score is traceable.
- The **World Bank Guidelines** (Policy Recommendation 2) explicitly state: *"CSPs should understand and be able to explain to regulators the way credit scoring is incorporated into their processes and the logic involved in its functioning."*

**For our project:**
- We will prioritize **Logistic Regression with Weight of Evidence (WoE)** as our primary model because its coefficients directly represent log-odds contributions, satisfying Basel II's interpretability requirements.
- All experiments will be tracked with **MLflow** to maintain versioned documentation of feature engineering, proxy variable construction, and hyperparameter tuning — meeting the documentation expectations of Basel II Article 144.

### 2. Without a direct "default" label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

**Why a proxy is necessary:**
The raw transaction data from Xente eCommerce contains **no historical loan performance record** — no "did this customer default?" label. Supervised learning requires a binary target. Therefore, we must engineer a **credible proxy** using behavioral data.

Following the **RFMS methodology validated by Huang et al. (2018)** in the *Statistica Sinica* paper (page 6-7):

- **Recency (R):** Time since last transaction — lower recency signals higher risk
- **Frequency (F):** Number of transactions — lower frequency signals higher risk
- **Monetary (M):** Total transaction value — lower monetary signals higher risk

Our proxy will define **High Risk (Bad)** as customers in the lowest RFM quantile, and **Low Risk (Good)** as those in the highest quantile.

**Business risks of proxy-based prediction:**

**Business risks of proxy-based prediction:**

**Risk 1: Proxy Misalignment**

- Description: An RFM-defined "bad" customer may actually be financially responsible (for example, saving cash or using alternative payment methods)

- Mitigation: Validate against business rules and implement expert override overlays

**Risk 2: Label Bias**

- Description: The model learns to predict the proxy signal rather than true default behavior

- Mitigation: Compare proxy-based predictions with post-launch default data and test on holdout sets

**Risk 3: Segmentation Error**

- Description: RFM threshold choices (for example, top 20% versus top 30%) materially change model outcomes

- Mitigation: Document quantile selection rationale and perform sensitivity analysis on threshold choices

**Risk 4: Temporal Drift**

- Description: Customer behavior patterns change over time due to seasonality, promotions, or economic conditions

- Mitigation: Implement regular proxy recalibration on a monthly or quarterly basis

**Risk 5: Adverse Selection**

- Description: The proxy might correlate with protected or business-sensitive attributes

- Mitigation: Audit proxy distribution across customer segments for fairness concerns

The HKMA (2020) whitepaper reinforces this approach, noting that for MSMEs with "insufficient financial and operating data for underwriting" (Part One, Section 1.3), "transactional data... can be used to predict trends and patterns in an MSME's revenues" (Part One, Section 2.2.1) — exactly the RFM-based logic we will apply.

### 3. What are the key trade-offs between a simple, interpretable model and a high-performance model in a regulated financial context?

**Logistic Regression + WoE (Interpretable Model)**

- Interpretability is high because coefficients directly show log-odds contribution per feature

- Regulatory acceptance is well-established and widely accepted by regulators

- Performance (AUC) is moderate, typically around 0.70 to 0.75

- Overfitting risk is low when using proper regularization

- Documentation burden is standard, requiring coefficient tables and WoE charts

**Gradient Boosting (High-Performance Model)**

- Interpretability is low because it requires SHAP or LIME for explanation and is considered a "black box"

- Regulatory acceptance requires additional validation and some jurisdictions restrict its use

- Performance (AUC) is higher, typically around 0.80 to 0.85

- Overfitting risk is high without careful hyperparameter tuning

- Documentation burden is heavy, requiring feature importance plots, SHAP summaries, and partial dependence plots

**Our Hybrid Strategy**

- Phase 1 (Initial Deployment): Logistic Regression + WoE to ensure Basel II compliance with full interpretability

- Phase 2 (Performance Optimization): XGBoost + SHAP explanations because higher AUC reduces default losses while SHAP provides explainability

- Decision Rule: If XGBoost achieves AUC improvement greater than 0.05 with SHAP explainability tools, we recommend hybrid deployment. Both models will be tracked in MLflow with hyperparameters, metrics, and artifacts.

**Our hybrid strategy:**

The **World Bank Guidelines** (Policy Recommendation 4) mandate that "credit scoring models should be subject to a model governance framework" that includes "conceptual soundness" and "regular reviews." Accordingly:

- **Phase 1 (Initial Deployment):** Logistic Regression + WoE — ensuring Basel II compliance with interpretability.
- **Phase 2 (Performance Optimization):** XGBoost with SHAP explanations — if AUC improvement > 0.05, we will document the trade-off and provide SHAP force plots for individual predictions.

The **HKMA** study (Part Three, Section 1.2) found that while "selected machine learning algorithms demonstrated different predictive power... generally all were able to make effective default predictions" — justifying our two-phase approach where the simpler model serves as the regulatory baseline.

**References used for this section:**

1. Huang, D., Zhou, J., & Wang, H. (2018). *RFMS Method for Credit Scoring Based on Bank Card Transaction Data*. Statistica Sinica, 28, 2903-2919.

2. Hong Kong Monetary Authority (2020). *Alternative Credit Scoring of Micro-, Small and Medium-sized Enterprises*. HKMA.

3. World Bank Group (2019). *Credit Scoring Approaches Guidelines*. World Bank.

4. Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II)*. BIS.