"""
Model Comparison: Forward Models + Optimization Methods
Compare GPR, SVR, XGBoost and DE, L-BFGS-B
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
print("=" * 60)
print("PART 1: FORWARD MODELS COMPARISON")
print("=" * 60)

df = pd.read_csv('data/raw/experiment_design_27points.csv')
X = df[['Proportion1', 'Proportion2', 'Temp_C', 'Pressure_bar']]
y = df.filter(like='Alpha_')

selected_hz = [f'Alpha_{i}Hz' for i in range(1, 101)]
y_selected = y[selected_hz]

print(f"Data points: {len(df)}, Features: {X.shape[1]}, Outputs: {y_selected.shape[1]}")

# ============================================================
# 2. Compare Forward Models
# ============================================================
test_cols = ['Alpha_1Hz', 'Alpha_25Hz', 'Alpha_50Hz', 'Alpha_75Hz', 'Alpha_100Hz']

forward_models = {
    'GPR': GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=5, alpha=0.1, normalize_y=True, random_state=42
    ),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10, gamma='scale'))
    ]),
    'XGBoost': XGBRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        reg_lambda=1.0, verbosity=0, random_state=42
    ),
}

print("\nCross-Validation Results (5-Fold):")
print("-" * 40)
forward_results = {}
for name, model in forward_models.items():
    scores = [np.mean(cross_val_score(model, X, y_selected[col], cv=5, scoring='r2')) for col in test_cols]
    forward_results[name] = np.mean(scores)
    print(f"  {name:10s}: R² = {forward_results[name]:.4f}")

best_forward = max(forward_results, key=forward_results.get)
print(f"\nBest Forward Model: {best_forward} (R² = {forward_results[best_forward]:.4f})")

# ============================================================
# 3. Train Best Model
# ============================================================
print(f"\nTraining {best_forward} for all frequencies...")

def create_model(name):
    if name == 'GPR':
        return GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
            n_restarts_optimizer=5, alpha=0.1, normalize_y=True, random_state=42
        )
    elif name == 'SVR':
        return Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10, gamma='scale'))])
    else:
        return XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, reg_lambda=1.0, verbosity=0, random_state=42)

trained_models = {}
for col in y_selected.columns:
    model = create_model(best_forward)
    model.fit(X, y_selected[col])
    trained_models[col] = model
print(f"Trained {len(trained_models)} models")

# ============================================================
# 4. Optimization Comparison
# ============================================================
print("\n" + "=" * 60)
print("PART 2: OPTIMIZATION METHODS COMPARISON")
print("=" * 60)

target_output = np.linspace(0.7, 0.98, len(selected_hz))
PROPORTION_SUM = 100

def predict_all(x4):
    x_arr = np.array(x4).reshape(1, -1)
    return np.array([trained_models[col].predict(x_arr)[0] for col in y_selected.columns])

def objective_3d(x3):
    x4 = [x3[0], PROPORTION_SUM - x3[0], x3[1], x3[2]]
    preds = predict_all(x4)
    return np.mean((preds - target_output) ** 2)

X_df = df[['Proportion1', 'Proportion2', 'Temp_C', 'Pressure_bar']]
p1_min = max(X_df['Proportion1'].min(), PROPORTION_SUM - X_df['Proportion2'].max())
p1_max = min(X_df['Proportion1'].max(), PROPORTION_SUM - X_df['Proportion2'].min())
bounds_3d = [(p1_min, p1_max), (X_df['Temp_C'].min(), X_df['Temp_C'].max()), (X_df['Pressure_bar'].min(), X_df['Pressure_bar'].max())]

opt_results = {}

# Method 1: Differential Evolution
print("\n[1] Differential Evolution...")
start = time.time()
result_de = differential_evolution(objective_3d, bounds_3d, seed=42, maxiter=100, popsize=10, polish=False, disp=False)
time_de = time.time() - start
opt_results['DE'] = {'mse': result_de.fun, 'time': time_de, 'x': result_de.x}
print(f"    MSE: {result_de.fun:.6f}, Time: {time_de:.2f}s")

# Method 2: L-BFGS-B Multi-start
print("[2] L-BFGS-B Multi-start...")
start = time.time()
best_mse, best_x = np.inf, None
np.random.seed(42)
for _ in range(20):
    x0 = [np.random.uniform(b[0], b[1]) for b in bounds_3d]
    result = minimize(objective_3d, x0, method='L-BFGS-B', bounds=bounds_3d)
    if result.fun < best_mse:
        best_mse, best_x = result.fun, result.x
time_lbfgs = time.time() - start
opt_results['L-BFGS-B'] = {'mse': best_mse, 'time': time_lbfgs, 'x': best_x}
print(f"    MSE: {best_mse:.6f}, Time: {time_lbfgs:.2f}s")

# Method 3: Hybrid
print("[3] DE + L-BFGS-B Hybrid...")
start = time.time()
result_hybrid = minimize(objective_3d, result_de.x, method='L-BFGS-B', bounds=bounds_3d)
time_hybrid = time.time() - start + time_de
opt_results['Hybrid'] = {'mse': result_hybrid.fun, 'time': time_hybrid, 'x': result_hybrid.x}
print(f"    MSE: {result_hybrid.fun:.6f}, Time: {time_hybrid:.2f}s")

# ============================================================
# 5. Final Summary
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"\nBest Forward Model: {best_forward} (R² = {forward_results[best_forward]:.4f})")

best_opt = min(opt_results, key=lambda x: opt_results[x]['mse'])
print(f"Best Optimization:  {best_opt} (MSE = {opt_results[best_opt]['mse']:.6f})")

x_best = opt_results[best_opt]['x']
print(f"\nOptimal Input:")
print(f"  Proportion1:  {x_best[0]:.4f}")
print(f"  Proportion2:  {PROPORTION_SUM - x_best[0]:.4f}")
print(f"  Temp_C:       {x_best[1]:.4f}")
print(f"  Pressure_bar: {x_best[2]:.4f}")

print("\n" + "=" * 60)
print("Done!")