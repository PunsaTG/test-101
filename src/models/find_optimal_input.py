import numpy as np
from scipy.optimize import differential_evolution
from xgboost import XGBRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# โหลดข้อมูล
df = pd.read_csv('material_100hz_data.csv')
X = df.iloc[:, :4]   # Proportion1, Proportion2, Temp_C, Pressure_bar
y = df.iloc[:, 4:]   # Alpha_1Hz ถึง Alpha_100Hz

print("🔄 กำลังเทรน XGBoost models สำหรับแต่ละ Hz...")
# เทรน XGBoost model สำหรับแต่ละ Hz
models = {}
for col in y.columns:
    model = XGBRegressor(random_state=1, n_estimators=100, verbosity=0)
    model.fit(X, y[col])
    models[col] = model
print("✅ เทรนเสร็จสิ้น!\n")

# =========================================
# ค่าที่ดีที่สุด: Alpha อยู่ในช่วง 0.7 - 1.0
# (ดูดซับเสียงได้ดีที่สุด)
# =========================================
OPTIMAL_MIN = 0.7
OPTIMAL_MAX = 1.0
OPTIMAL_TARGET = 0.85  # เป้าหมายกลางช่วง

def objective_maximize_absorption(x):
    """หาค่า input ที่ทำให้ Alpha สูงที่สุด (ใกล้ 0.85 หรือสูงกว่า)"""
    predictions = []
    for col in y.columns:
        pred = models[col].predict([x])[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    
    # คำนวณ score:
    # 1. ลงโทษถ้าต่ำกว่า 0.7 มาก
    # 2. ให้รางวัลถ้าอยู่ในช่วง 0.7-1.0
    penalty = 0
    for pred in predictions:
        if pred < OPTIMAL_MIN:
            penalty += (OPTIMAL_MIN - pred) ** 2 * 10  # ลงโทษหนักถ้าต่ำกว่า 0.7
        elif pred > OPTIMAL_MAX:
            penalty += (pred - OPTIMAL_MAX) ** 2  # ลงโทษเบาถ้าเกิน 1.0
    
    # ลบค่าเฉลี่ย (เพื่อ maximize) + penalty
    score = -np.mean(predictions) + penalty
    return score

# กำหนดขอบเขตของ input (min, max จากข้อมูลจริง)
bounds = [
    (X['Proportion1'].min(), X['Proportion1'].max()),   # Proportion1
    (X['Proportion2'].min(), X['Proportion2'].max()),   # Proportion2  
    (X['Temp_C'].min(), X['Temp_C'].max()),             # Temp_C
    (X['Pressure_bar'].min(), X['Pressure_bar'].max()), # Pressure_bar
]

print("🔍 กำลังค้นหาค่า Input ที่ดีที่สุด...")
print(f"   เป้าหมาย: Alpha ทุก Hz อยู่ในช่วง {OPTIMAL_MIN} - {OPTIMAL_MAX}\n")

# ใช้ Differential Evolution (global optimization) เพื่อหาค่าที่ดีที่สุด
result = differential_evolution(
    objective_maximize_absorption, 
    bounds, 
    seed=42,
    maxiter=200,
    tol=1e-6,
    disp=True  # แสดง progress
)

# คำนวณ predictions สำหรับค่าที่ดีที่สุด
final_predictions = []
for col in y.columns:
    pred = models[col].predict([result.x])[0]
    final_predictions.append(pred)
final_predictions = np.array(final_predictions)

# นับจำนวนที่อยู่ในช่วง optimal
in_optimal_range = np.sum((final_predictions >= OPTIMAL_MIN) & (final_predictions <= OPTIMAL_MAX))
below_optimal = np.sum(final_predictions < OPTIMAL_MIN)
above_optimal = np.sum(final_predictions > OPTIMAL_MAX)

# แสดงผลลัพธ์
print("=" * 60)
print("🎯 ค่า INPUT ที่ดีที่สุดสำหรับการดูดซับเสียง")
print("=" * 60)
print(f"  📊 Proportion1:   {result.x[0]:.4f}")
print(f"  📊 Proportion2:   {result.x[1]:.4f}")
print(f"  🌡️  Temp_C:        {result.x[2]:.4f}")
print(f"  ⚡ Pressure_bar:  {result.x[3]:.4f}")
print("=" * 60)
print("\n📈 ผลการวิเคราะห์ค่า Alpha:")
print(f"   ค่าต่ำสุด:  {final_predictions.min():.4f}")
print(f"   ค่าสูงสุด:  {final_predictions.max():.4f}")
print(f"   ค่าเฉลี่ย:  {final_predictions.mean():.4f}")
print("\n📊 สรุปช่วงค่า:")
print(f"   ✅ อยู่ในช่วง 0.7-1.0 (ดีมาก):  {in_optimal_range}/100 Hz ({in_optimal_range}%)")
print(f"   ⚠️  ต่ำกว่า 0.7:              {below_optimal}/100 Hz")
print(f"   ⚠️  สูงกว่า 1.0:              {above_optimal}/100 Hz")
print("=" * 60)

# แสดงตัวอย่าง predictions
print("\n📋 ตัวอย่างค่า Alpha ที่ได้ (ทุกๆ 10 Hz):")
print("-" * 40)
for i in range(0, 100, 10):
    status = "✅" if OPTIMAL_MIN <= final_predictions[i] <= OPTIMAL_MAX else "⚠️"
    print(f"   {i+1:3d} Hz: {final_predictions[i]:.4f} {status}")
print("-" * 40)

# แสดงค่า Alpha ทั้งหมด
print("\n📋 ค่า Alpha ทั้งหมด (100 Hz):")
print("-" * 60)
for i, col in enumerate(y.columns):
    status = "✅" if OPTIMAL_MIN <= final_predictions[i] <= OPTIMAL_MAX else "⚠️"
    if (i + 1) % 5 == 0 or i == 0:  # แสดงทุก 5 Hz
        print(f"   {col}: {final_predictions[i]:.4f} {status}")
