
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Create output directory
output_dir = "maritime_gp_analysis_FIXED"
os.makedirs(output_dir, exist_ok=True)

print("="*60)

# ---------------------------
# Step 1: Synthetic data generation
# ---------------------------
np.random.seed(42)
num_vessels = 40
num_quarters = 24
vessels = [f"Asset{i+1}" for i in range(num_vessels)]
quarters = np.arange(1, num_quarters + 1)

# More realistic base parameters
base_vessel_age = np.random.uniform(5, 20, num_vessels)
base_time_since_drydock = np.random.uniform(0.5, 4.5, num_vessels)
base_vessel_size = np.random.choice(['Small', 'Medium', 'Large'], num_vessels, p=[0.3, 0.5, 0.2])
vessel_size_multiplier = {'Small': 0.8, 'Medium': 1.0, 'Large': 1.3}
size_multipliers = [vessel_size_multiplier[size] for size in base_vessel_size]

print(f"Generated {num_vessels} vessels across {num_quarters} quarters")

# Initialize lists
vals, ages, drydocks, waccs, opexs, inflations = [], [], [], [], [], []
freights, lagged_bdis, lagged_values = [], [], []
covid, imoregs, geopol = [], [], []
vessel_ids = []

# Macro variables
base_inflation = 2.5
inflation_trend = base_inflation + 0.1 * np.sin(np.linspace(0, 2*np.pi, num_quarters)) + np.random.normal(0, 0.2, num_quarters)
base_freight = 1500
seasonal_freight = 150 * np.sin(np.linspace(0, 4*np.pi, num_quarters))
trend_freight = 20 * (quarters - 12)
noise_freight = np.random.normal(0, 50, num_quarters)
forward_freight_rate = base_freight + seasonal_freight + trend_freight + noise_freight
base_bdi = 1200
bdi_trend = 100 * np.sin(np.linspace(0, 1.5*np.pi, num_quarters))
bdi_noise = np.random.normal(0, 80, num_quarters)
bdi = base_bdi + bdi_trend + bdi_noise

# COVID impact
covid_intensity = np.zeros(num_quarters)
for q in quarters:
    if 8 <= q <= 14:
        if q <= 11:
            covid_intensity[q-1] = 0.3 * (q - 7) / 4
        else:
            covid_intensity[q-1] = 0.3 * (15 - q) / 4

# IMO regulations
imo_intensity = np.zeros(num_quarters)
for q in quarters:
    if q >= 12:
        imo_intensity[q-1] = min(0.8, 0.2 * (q - 11))
# Geopolitical events
geopol_events = np.random.choice([0, 1], size=num_quarters, p=[0.92, 0.08])
print("Generated realistic macro variables")

# Vessel-specific data generation
for vessel_i, vessel in enumerate(vessels):
    vessel_age = base_vessel_age[vessel_i]
    drydock = base_time_since_drydock[vessel_i]
    size_mult = size_multipliers[vessel_i]
    base_value = 35 + 10 * size_mult
    time_trend = np.linspace(0, 3, num_quarters)
    vessel_volatility = np.random.uniform(0.5, 1.5)
    vessel_noise = np.random.normal(0, vessel_volatility, num_quarters)
    vessel_base_values = base_value + time_trend + vessel_noise
    # WACC AR(1)
    base_wacc = np.random.normal(7.5, 0.5)
    vessel_waccs = [base_wacc]
    for t in range(1, num_quarters):
        new_wacc = 0.85 * vessel_waccs[-1] + 0.15 * base_wacc + np.random.normal(0, 0.1)
        new_wacc = max(5.0, min(10.0, new_wacc))
        vessel_waccs.append(new_wacc)
    for t in range(num_quarters):
        current_age = vessel_age + 0.25 * t
        current_drydock = max(0, drydock + 0.25 * t)
        if t > 0 and t % 20 == 0:
            current_drydock = 0
        vessel_ids.append(vessel_i)
        ages.append(current_age)
        drydocks.append(current_drydock)
        waccs.append(vessel_waccs[t])
        base_opex = 4000 + 200 * size_mult
        age_effect = 50 * max(0, current_age - 10)
        seasonal_opex = 200 * np.cos(2 * np.pi * t / 4)
        opex = base_opex + age_effect + seasonal_opex + np.random.normal(0, 100)
        opexs.append(max(3000, opex))
        inflations.append(inflation_trend[t])
        freights.append(forward_freight_rate[t])
        lagged_bdis.append(bdi[t])
        lagged_values.append(vessel_base_values[t])
        covid.append(covid_intensity[t])
        imoregs.append(imo_intensity[t])
        geopol.append(geopol_events[t])
                # Calculate the vessel's base valuation from its base value, time trend, and noise:
        valuation = vessel_base_values[t]

        # Compute additional effects
        age_depreciation = -0.3 * max(0, current_age - 15)
        drydock_effect = -0.5 * max(0, current_drydock - 3)
        freight_effect = 0.002 * (forward_freight_rate[t] - base_freight)
        bdi_effect = 0.001 * (bdi[t] - base_bdi)
        covid_effect = -8 * covid_intensity[t]
        imo_effect = 2 * imo_intensity[t]
        geopol_effect = -3 * geopol_events[t] if geopol_events[t] else 0

        # Incorporate WACC by computing a discount factor.
        # Here, we set a target or baseline WACC of 7.5.
        # The adjustment factor (0.05 in this example) determines how sensitive the discount is to deviations from the target.
        base_wacc_target = 7.5
        adjustment_factor = 0.05
        discount_factor = 1 / (1 + (vessel_waccs[t] - base_wacc_target) * adjustment_factor)
        # If vessel_waccs[t] > 7.5, discount_factor < 1, lowering the valuation, and vice versa.

        # Now combine all the effects, add a small random noise, and apply the discount factor.
        final_valuation = (valuation + age_depreciation + drydock_effect +
                           freight_effect + bdi_effect + covid_effect +
                           imo_effect + geopol_effect + np.random.normal(0, 0.8)) * discount_factor

        vals.append(final_valuation)

synthetic_df = pd.DataFrame({
    'Asset': np.repeat(vessels, num_quarters),
    'VesselID': vessel_ids,
    'Quarter': np.tile(quarters, num_vessels),
    'Valuation': vals,
    'VesselAge': ages,
    'TimeSinceDryDock': drydocks,
    'WACC': waccs,
    'Opex': opexs,
    'Inflation': inflations,
    'ForwardFreightRate': freights,
    'LaggedBDI': lagged_bdis,
    'LaggedVesselValue': lagged_values,
    'COVID': covid,
    'IMORegs': imoregs,
    'Geopolitical': geopol
})

print(f"Generated realistic synthetic data: {synthetic_df.shape}")
print(f"Valuation range: ${synthetic_df['Valuation'].min():.1f}M to ${synthetic_df['Valuation'].max():.1f}M")



# ---------------------------
# Check Correlations Between WACC and Other Features
# ---------------------------
# %% Diagnostic Analysis: Correlation Matrix and Heatmap


features_to_check = ['WACC', 'LaggedVesselValue', 'ForwardFreightRate',
                     'LaggedBDI', 'COVID', 'IMORegs', 'Geopolitical',
                     'VesselAge', 'TimeSinceDryDock', 'Opex', 'Inflation', 'Valuation']
corr_matrix = synthetic_df[features_to_check].corr()
print("Correlation Matrix:")
print(corr_matrix)
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Including Final Valuation")
plt.show()

# %% Diagnostic Analysis: Selected Feature Importance
from sklearn.feature_selection import f_regression
# Assume selected_features is defined after your SelectKBest block
# For illustration purpose, assume:
selected_features = ['WACC', 'ForwardFreightRate', 'LaggedVesselValue', 'COVID', 'IMORegs', 'Geopolitical']
forced_scores, _ = f_regression(synthetic_df[selected_features], synthetic_df['Valuation'])
print("Recalculated F-Scores for forced features:", forced_scores)
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(selected_features)), forced_scores, color='skyblue')
plt.yticks(range(len(selected_features)), selected_features, fontsize=12)
plt.xlabel('F-Score', fontsize=12)
plt.title('Selected Feature Importance', fontsize=14)
plt.grid(axis='x', alpha=0.3)
for i, score in enumerate(forced_scores):
    plt.text(score + 1, i, f"{score:.2f}", va='center', fontsize=10)
plt.show()


# ---------------------------
# Step 2:  model 
# ---------------------------

# 1) Train/test pools
train_vessels = vessels[:25]
test_vessels = vessels[25:30]
print(f"   Training vessels: {len(train_vessels)}, Test vessels: {len(test_vessels)}")

print("Feature selection - using most important features")
all_features = ['Quarter', 'VesselAge', 'TimeSinceDryDock', 'WACC', 'Opex',
                'Inflation', 'ForwardFreightRate', 'LaggedBDI', 'LaggedVesselValue',
                'COVID', 'IMORegs', 'Geopolitical']
train_data = synthetic_df[synthetic_df['Asset'].isin(train_vessels)].copy()
X_all = train_data[all_features]
y_all = train_data['Valuation']

# Run SelectKBest to choose top k features; here k=6 as before.
selector = SelectKBest(f_regression, k=6)
_ = selector.fit_transform(X_all, y_all)
selected_features = [all_features[i] for i in selector.get_support(indices=True)]

# Force inclusion of 'WACC'
if 'WACC' not in selected_features:
    selected_features.append('WACC')

print(f"   Selected features (forced to include WACC): {selected_features}")

# Re-calculate feature scores for the forced feature set
from sklearn.feature_selection import f_regression
# Note: Using X_all[selected_features] ensures correct dimensionality.
forced_scores, _ = f_regression(X_all[selected_features], y_all)
print("   Recalculated F-Scores for forced features:", forced_scores)



# 3) Polynomial time feature
print("Adding polynomial time features")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
_ = poly.fit_transform(train_data[['Quarter']].values)  # For structure mirroring

# One-hot encode vessels (IMPORTANT: use sparse_output for sklearn ≥1.2)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
vessel_ids_encoded = encoder.fit_transform(train_data[['Asset']])

# Build enhanced training features:
# Use the manually selected features (which now includes WACC),
# add an extra feature (Quarter^2), then concatenate vessel one-hots.
train_data_enhanced = train_data[selected_features].copy()
train_data_enhanced['Quarter^2'] = train_data['Quarter'] ** 2
vessel_features = pd.DataFrame(
    vessel_ids_encoded,
    index=train_data.index,
    columns=encoder.get_feature_names_out(['Asset'])
)
train_data_enhanced = pd.concat([train_data_enhanced, vessel_features], axis=1)

feature_columns = list(train_data_enhanced.columns)
n_vessels_encoded = len(encoder.categories_[0])


# 4) Scale features and target
print(" feature scaling")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data_enhanced)
y_scaled = (y_all - y_all.mean()) / y_all.std()
y_mean, y_std = y_all.mean(), y_all.std()
print(f"   Features used: {len(feature_columns)}")
print(f"   Training samples: {len(X_scaled)}")

# ---------------------------------------------------
# Using vessel-aware kernel 
def create_vessel_kernel(n_features, n_vessels):
    """Kernel with vessel-level bias + smooth time features."""
    base_kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    vessel_kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    return base_kernel + vessel_kernel + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 1.0))

# ---------------------------------------------------
# Proper time series train/test split 
train_indices, test_indices = [], []
for vessel in train_vessels:
    mask = (train_data['Asset'] == vessel)
    vessel_data_idx = train_data[mask].index
    n_points = len(vessel_data_idx)
    split_point = int(0.7 * n_points)
    train_indices.extend(vessel_data_idx[:split_point])
    test_indices.extend(vessel_data_idx[split_point:])
X_train = X_scaled[train_data_enhanced.index.isin(train_indices)]
X_test  = X_scaled[train_data_enhanced.index.isin(test_indices)]
y_train = y_scaled[train_data_enhanced.index.isin(train_indices)]
y_test  = y_scaled[train_data_enhanced.index.isin(test_indices)]
print(f"   Final train size: {len(X_train)}, test size: {len(X_test)}")

# ---------------------------------------------------
#  Training  model 
print("Training  model")
gp = GaussianProcessRegressor(
    kernel=create_vessel_kernel(len(feature_columns), n_vessels_encoded),
    n_restarts_optimizer=20,
    alpha=1e-8,
    normalize_y=False,
    random_state=42
)
print("   Training GP model...")
gp.fit(X_train, y_train)


# ---------------------------
#  Hold-Out Calibration
# ---------------------------
# Partition a hold-out calibration set from X_train and y_train (example, 20% of the training samples)

from sklearn.model_selection import train_test_split

# Split X_train into a reduced training set and a calibration set
X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# Retrain the model on the reduced training set for calibration
gp_calib = GaussianProcessRegressor(
    kernel=create_vessel_kernel(len(feature_columns), n_vessels_encoded),
    n_restarts_optimizer=20,
    alpha=1e-8,
    normalize_y=False,
    random_state=42
)
gp_calib.fit(X_train_sub, y_train_sub)

# Generate predictions on the calibration set
y_calib_pred, y_calib_std = gp_calib.predict(X_calib, return_std=True)

# Define candidate calibration factors and nominal coverage
candidate_factors = np.linspace(0.5, 1.5, 11)
nominal_coverage = 0.95
z_score = 1.96

best_factor = None
best_coverage_diff = np.inf

for factor in candidate_factors:
    y_calib_std_calibrated = y_calib_std * factor
    lower_bound = y_calib_pred - z_score * y_calib_std_calibrated
    upper_bound = y_calib_pred + z_score * y_calib_std_calibrated
    # Note: y_calib is in the scaled domain - ensure consistency
    coverage = np.mean((y_calib >= lower_bound) & (y_calib <= upper_bound))
    diff = abs(coverage - nominal_coverage)
    if diff < best_coverage_diff:
        best_coverage_diff = diff
        best_factor = factor

print(" Holdout Calibration:")
print(f"Best calibration factor from hold-out set: {best_factor:.3f}")

# ---------------------------
# Continue with Final Predictions Using the Calibrated Uncertainties
# ---------------------------
print("Making predictions with calibrated uncertainty on Test Set...")
y_pred_scaled, y_std_scaled = gp.predict(X_test, return_std=True)
# Unscale predictions and uncertainties as needed:
y_pred = y_pred_scaled * y_std + y_mean
y_test_unscaled = y_test * y_std + y_mean
# Multiply the calibration factor with a manual adjustment (here 1.10) if desired:
y_std_unscaled = y_std_scaled * y_std * best_factor * 1.2  # apply calibration factor manually

# Compute prediction intervals
interval_lower = y_pred - z_score * y_std_unscaled
interval_upper = y_pred + z_score * y_std_unscaled

# Calculate PICP (coverage)
PICP_calibrated = np.mean((y_test_unscaled >= interval_lower) & (y_test_unscaled <= interval_upper))
print("Calibrated Prediction Interval Coverage Probability (PICP):", PICP_calibrated)

# Calculate and print the average interval width (sharpness)
precision = np.mean(interval_upper - interval_lower)
print("Average Interval Width (Sharpness) after calibration:", precision)

#-------------------------------------------------------------------------------------------------------------
# --- Integrate AIC and BIC calculation ---
# 1. Extract log marginal likelihood
log_marginal_likelihood = gp.log_marginal_likelihood(gp.kernel_.theta)
print("Log Marginal Likelihood:", log_marginal_likelihood)

# 2. Compute effective number of parameters, k_eff
K = gp.kernel_(X_train)  # Kernel matrix on training data
n = X_train.shape[0]
sigma2 = gp.alpha  # Noise level used in GP (alpha)
K_noisy = K + sigma2 * np.eye(n)
sol = np.linalg.solve(K_noisy, K)
k_eff = np.trace(sol)
print("Effective number of parameters (k_eff):", k_eff)

# 3. Calculate AIC and BIC
aic = -2 * log_marginal_likelihood + 2 * k_eff
bic = -2 * log_marginal_likelihood + k_eff * np.log(n)
print("AIC:", aic)
print("BIC:", bic)
# --- End AIC/BIC ---

print(" Making predictions...")
y_pred_scaled, y_std_scaled = gp.predict(X_test, return_std=True)
# Unscale predictions
y_pred = y_pred_scaled * y_std + y_mean
y_test_unscaled = y_test * y_std + y_mean
y_std_unscaled = y_std_scaled * y_std*1.2

# ----- Basic Risk-Tiered Review Protocols -----
green_thresh = 1.0  # Green: z < 1.0
amber_thresh = 2.0  # Amber: 1 <= z < 2
risk_categories = []
z_scores = []  # storing computed z-scores for each observation

for i, (actual, pred, std_val) in enumerate(zip(y_test_unscaled, y_pred, y_std_unscaled)):
    z = abs(actual - pred) / std_val if std_val > 1e-6 else 0.0
    z_scores.append(z)
    if z < green_thresh:
        risk = 'Green'
    elif z < amber_thresh:
        risk = 'Amber'
    else:
        risk = 'Red'
    risk_categories.append(risk)

from collections import Counter
risk_counts = Counter(risk_categories)
print("Risk Tiered Review Summary for Test Set:")
for category, count in risk_counts.items():
    print(f"{category}: {count} observations")

risk_df = pd.DataFrame({
    'Actual': y_test_unscaled,
    'Prediction': y_pred,
    'Std': y_std_unscaled,
    'Z-score': z_scores,
    'Risk': risk_categories
})
print("\nSample of Risk Classification Results:")
print(risk_df.head())

# Metrics
mae_test = mean_absolute_error(y_test_unscaled, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
r2 = r2_score(y_test_unscaled, y_pred)
print(f"Test MAE:  {mae_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")
print(f"R² Score:  {r2:.4f}")
print(f"Mean uncertainty: ±{np.mean(y_std_unscaled):.4f}")


if r2 > 0:
    print("Positive R² - model is learning!")
    if r2 > 0.5:
        print(" R² > 0.5 - model performings well")
    elif r2 > 0.3:
        print("R² > 0.3 - reasonable performance")
    else:
        print("Positive R² but not good performance")
else:
    print("negative R² - need more adjustments")


# ---------------------------
# Step 7.5: Model Comparison Among Different Approaches
# ---------------------------

# ---------------------------
# For GP Variant 1: Baseline Vessel-Aware Kernel
# ---------------------------
gp_baseline = GaussianProcessRegressor(
    kernel=create_vessel_kernel(len(feature_columns), n_vessels_encoded),  # current vessel-aware kernel
    n_restarts_optimizer=20,
    alpha=1e-8,
    normalize_y=False,
    random_state=42
)
gp_baseline.fit(X_train, y_train)
y_pred_gp_baseline_scaled = gp_baseline.predict(X_test)
# Unscale predictions for baseline GP
y_pred_gp_baseline = y_pred_gp_baseline_scaled * y_std + y_mean
# Unscale y_test (if not already done)
y_test_unscaled = y_test * y_std + y_mean
r2_baseline = r2_score(y_test_unscaled, y_pred_gp_baseline)
mae_baseline = mean_absolute_error(y_test_unscaled, y_pred_gp_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_gp_baseline))


# ---------------------------
# For GP Variant 2: GP with Matern Kernel
# ---------------------------
def create_vessel_kernel_matern(n_features, n_vessels):
    """
    Create a vessel-aware kernel where:
    - Base kernel (global effects) is kept as Constant * RBF.
    - Vessel-specific kernel is Constant * Matern(nu=2.5).
    - A WhiteKernel is added for noise.
    """
    base_kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
    vessel_kernel = C(1.0, (0.1, 10.0)) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5)
    return base_kernel + vessel_kernel + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 1.0))

gp_matern = GaussianProcessRegressor(
    kernel=create_vessel_kernel_matern(len(feature_columns), n_vessels_encoded),
    n_restarts_optimizer=20,
    alpha=1e-8,
    normalize_y=False,
    random_state=42
)
gp_matern.fit(X_train, y_train)
y_pred_gp_matern_scaled = gp_matern.predict(X_test)
# Unscale predictions for GP with Matern variant
y_pred_gp_matern = y_pred_gp_matern_scaled * y_std + y_mean

# Compute error metrics on unscaled predictions
r2_matern = r2_score(y_test_unscaled, y_pred_gp_matern)
mae_matern = mean_absolute_error(y_test_unscaled, y_pred_gp_matern)
rmse_matern = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_gp_matern))

# --- Compute AIC and BIC for GP with Matern variant ---
log_marginal_likelihood_matern = gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta)
K_matern = gp_matern.kernel_(X_train)
n_train = X_train.shape[0]
sigma2 = gp_matern.alpha  # Noise level from GP
K_noisy_matern = K_matern + sigma2 * np.eye(n_train)
sol_matern = np.linalg.solve(K_noisy_matern, K_matern)
k_eff_matern = np.trace(sol_matern)
aic_matern = -2 * log_marginal_likelihood_matern + 2 * k_eff_matern
bic_matern = -2 * log_marginal_likelihood_matern + k_eff_matern * np.log(n_train)

print("GP with Matern (nu=2.5) for Vessel Component:")
print(f"   R²: {r2_matern:.3f}, MAE: {mae_matern:.3f}, RMSE: {rmse_matern:.3f}")
print(f"   AIC: {aic_matern:.3f}, BIC: {bic_matern:.3f}")


# ---------------------------
# Linear Regression Model
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr_scaled = lr_model.predict(X_test)
# Unscale predictions for Linear Regression
y_pred_lr = y_pred_lr_scaled * y_std + y_mean
r2_lr = r2_score(y_test_unscaled, y_pred_lr)
mae_lr = mean_absolute_error(y_test_unscaled, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_lr))


# ---------------------------
# Random Forest Model
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf_scaled = rf_model.predict(X_test)
# Unscale predictions for Random Forest
y_pred_rf = y_pred_rf_scaled * y_std + y_mean
r2_rf = r2_score(y_test_unscaled, y_pred_rf)
mae_rf = mean_absolute_error(y_test_unscaled, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_rf))


# ----- Comparison Metrics -----
print("\n--- Model Comparison Results ---")
print("Baseline GP:")
print(f"   R²: {r2_baseline:.3f}, MAE: {mae_baseline:.3f}, RMSE: {rmse_baseline:.3f}")
print("GP with Matern (nu=2.5) for Vessel Component:")
print(f"   R²: {r2_matern:.3f}, MAE: {mae_matern:.3f}, RMSE: {rmse_matern:.3f}")
print("Linear Regression:")
print(f"   R²: {r2_lr:.3f}, MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}")
print("Random Forest:")
print(f"   R²: {r2_rf:.3f}, MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}")


# ---------------------------
# Step 3: Visualization
# ---------------------------
print("\n Creating visualizations.")
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Feature importance (modified to use forced_scores)
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(range(len(selected_features)), forced_scores)
ax1.set_yticks(range(len(selected_features)))
ax1.set_yticklabels(selected_features)
ax1.set_xlabel('F-Score')
ax1.set_title('Selected Feature Importance')
ax1.grid(True, alpha=0.3)
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

# 2. Learning curve (using vessel-aware kernel)
ax2 = fig.add_subplot(gs[0, 1])
sample_sizes = np.linspace(50, len(X_train), 10, dtype=int)
train_scores, test_scores = [], []
for size in sample_sizes:
    gp_temp = GaussianProcessRegressor(
        kernel=create_vessel_kernel(len(feature_columns), n_vessels_encoded),
        n_restarts_optimizer=10,
        random_state=42
    )
    gp_temp.fit(X_train[:size], y_train[:size])
    train_pred = gp_temp.predict(X_train[:size])
    test_pred = gp_temp.predict(X_test)
    train_scores.append(r2_score(y_train[:size], train_pred))
    test_scores.append(r2_score(y_test, test_pred))
ax2.plot(sample_sizes, train_scores, 'o-', label='Training R²', color='blue')
ax2.plot(sample_sizes, test_scores, 's-', label='Test R²', color='red')
ax2.set_xlabel('Training Set Size')
ax2.set_ylabel('R² Score')
ax2.set_title('Learning Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Prediction vs Actual Scatter Plot with Error Bars
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(y_test_unscaled, y_pred, alpha=0.6, c=y_std_unscaled, cmap='viridis', s=50)
min_val = min(y_test_unscaled.min(), y_pred.min())
max_val = max(y_test_unscaled.max(), y_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
# Add error bars:
ax3.errorbar(y_test_unscaled, y_pred, yerr=z_score * y_std_unscaled, fmt='none', ecolor='gray', alpha=0.5)
ax3.set_xlabel('Actual Valuation (Million USD)')
ax3.set_ylabel('Predicted Valuation (Million USD)')
ax3.set_title(f'Predictions vs Actual (R² = {r2:.3f})')
plt.colorbar(scatter, ax=ax3, label='Prediction Uncertainty')
ax3.grid(True, alpha=0.3)

# 4. Residual Analysis
ax4 = fig.add_subplot(gs[1, 0])
residuals = y_test_unscaled - y_pred
ax4.scatter(y_pred, residuals, alpha=0.6, color='coral')
ax4.axhline(y=0, color='r', linestyle='--')
ax4.set_xlabel('Predicted Valuation')
ax4.set_ylabel('Residuals')
ax4.set_title('Residual Analysis')
ax4.grid(True, alpha=0.3)

# 5. Time Series Example 
ax5 = fig.add_subplot(gs[1, 1:])
test_vessel = None
for vessel in train_vessels:
    vessel_test_data = synthetic_df[synthetic_df['Asset'] == vessel].iloc[int(0.7*24):]
    if len(vessel_test_data) > 5:
        test_vessel = vessel
        break
if test_vessel:
    vessel_data = synthetic_df[synthetic_df['Asset'] == test_vessel].copy()
    vessel_enhanced = vessel_data[selected_features].copy()
    vessel_enhanced['Quarter^2'] = vessel_data['Quarter'] ** 2
    vfeat = encoder.transform(vessel_data[['Asset']])
    vfeat_df = pd.DataFrame(vfeat, index=vessel_data.index, columns=encoder.get_feature_names_out(['Asset']))
    vessel_enhanced = pd.concat([vessel_enhanced, vfeat_df], axis=1)
    vessel_enhanced = vessel_enhanced[feature_columns]
    vessel_scaled = scaler.transform(vessel_enhanced)
    vessel_pred_scaled, vessel_std_scaled = gp.predict(vessel_scaled, return_std=True)
    vessel_pred = vessel_pred_scaled * y_std + y_mean
    # Use the calibrated uncertainty adjustment for consistency:
    vessel_uncertainty = vessel_std_scaled * y_std * best_factor
    ax5.plot(vessel_data['Quarter'], vessel_data['Valuation'], 'b-', linewidth=2, label='Actual', alpha=0.8)
    ax5.plot(vessel_data['Quarter'], vessel_pred, 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax5.fill_between(vessel_data['Quarter'],
                     vessel_pred - z_score * vessel_uncertainty,
                     vessel_pred + z_score * vessel_uncertainty,
                     alpha=0.3, color='red', label='Uncertainty Band')
    split_quarter = vessel_data['Quarter'].iloc[int(0.7 * len(vessel_data))]
    ax5.axvline(x=split_quarter, color='green', linestyle=':', alpha=0.7, label='Train/Test Split')
    ax5.set_xlabel('Quarter')
    ax5.set_ylabel('Valuation (Million USD)')
    ax5.set_title(f'Time Series Prediction - {test_vessel}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

# 6. Diagnostics Panel (with AIC, BIC, and other performance metrics)
ax6 = fig.add_subplot(gs[2, :])
diagnostics = {
    'Mean Absolute Error': mae_test,
    'Root Mean Square Error': rmse_test,
    'R² Score': r2,
    'Mean Prediction Uncertainty': np.mean(y_std_unscaled),
    'Kernel Log-Likelihood': gp.log_marginal_likelihood(),
    'AIC': aic,
    'BIC': bic
}
metrics = list(diagnostics.keys())
values = list(diagnostics.values())
normalized_values = []
for metric, value in diagnostics.items():
    if metric in ['Mean Absolute Error', 'Root Mean Square Error', 'Mean Prediction Uncertainty']:
        normalized_values.append(max(0, 1 - value/5))
    elif metric == 'R² Score':
        normalized_values.append(max(0, min(1, value)))
    else:
        normalized_values.append(min(1, value/max(values)))
bars = ax6.barh(metrics, normalized_values, color='skyblue', edgecolor='navy', alpha=0.7)
ax6.set_xlabel('Normalized Score (Higher = Better)')
ax6.set_title('Model Performance Diagnostics')
ax6.grid(True, alpha=0.3)
for bar, value in zip(bars, values):
    ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', va='center', fontsize=10)

plt.suptitle('FIXED Maritime GP Model - Comprehensive Analysis', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------
# Step 4: Cross-validation with multiple vessels
# ---------------------------
group_kfold = GroupKFold(n_splits=5)
print(f"Cross-validation across vessel groups.")
cv_results = []
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_scaled, groups=train_data['Asset']), 1):
    X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
    y_train_cv, y_val_cv = y_scaled[train_idx], y_scaled[val_idx]
    gp_cv = GaussianProcessRegressor(
        kernel=create_vessel_kernel(len(feature_columns), len(encoder.categories_[0])),
        n_restarts_optimizer=5,
        alpha=1e-8,
        normalize_y=False,
        random_state=42
    )
    gp_cv.fit(X_train_cv, y_train_cv)
    y_val_pred = gp_cv.predict(X_val_cv)
    r2_cv = r2_score(y_val_cv, y_val_pred)
    print(f"Fold {fold}: Validation R² = {r2_cv:.3f}")
    cv_results.append({"Fold": fold, "R2": r2_cv})

if cv_results:
    mean_cv_r2 = np.mean([r["R2"] for r in cv_results])
    std_cv_r2 = np.std([r["R2"] for r in cv_results])
else:
    mean_cv_r2, std_cv_r2 = float("nan"), float("nan")

# ---------------------------
# Step 5: Results and summary
# ---------------------------
synthetic_df.to_csv(f'{output_dir}/fixed_synthetic_data.csv', index=False)
summary = f"""


SELECTED FEATURES:
{', '.join(feature_columns)}

PERFORMANCE:
- Test MAE:  {mae_test:.4f}
- Test RMSE: {rmse_test:.4f}
- R² Score:  {r2:.4f}
- Mean Uncertainty: ±{np.mean(y_std_unscaled):.4f}
- Kernel Log-Likelihood: {gp.log_marginal_likelihood():.3f}
- AIC: {aic:.3f}
- BIC: {bic:.3f}

TRAINING DETAILS:
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features used: {len(feature_columns)}
- Vessels in training: {len(train_vessels)}

CROSS-VALIDATION RESULTS:
- Average R²: {mean_cv_r2:.4f} ± {std_cv_r2:.4f}
"""

with open(f'{output_dir}/comprehensive_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(summary)


import seaborn as sns
import matplotlib.pyplot as plt

features_to_check = ['WACC', 'LaggedVesselValue', 'ForwardFreightRate',
                     'LaggedBDI', 'COVID', 'IMORegs', 'Geopolitical',
                     'VesselAge', 'TimeSinceDryDock', 'Opex', 'Inflation', 'Valuation']

corr_matrix = synthetic_df[features_to_check].corr()
print("Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Including Final Valuation")
plt.show()
