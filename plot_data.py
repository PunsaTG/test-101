import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# โหลดข้อมูล
df = pd.read_csv('material_100hz_data.csv')

# แยก Features และ Alpha values
features = df.iloc[:, :4]  # Proportion1, Proportion2, Temp_C, Pressure_bar
alpha_values = df.iloc[:, 4:]  # Alpha_1Hz ถึง Alpha_100Hz

# สร้างแกน X (1-100 Hz)
hz = np.arange(1, 101)

# สร้าง Figure ขนาดใหญ่
plt.figure(figsize=(14, 8))

# Plot แต่ละแถวของข้อมูล
for idx, row in df.iterrows():
    label = f"P1:{row['Proportion1']}, P2:{row['Proportion2']}, T:{row['Temp_C']}°C, Pr:{row['Pressure_bar']}bar"
    alpha_vals = row.iloc[4:].values.astype(float)
    plt.plot(hz, alpha_vals, label=label, linewidth=1.5, alpha=0.7)

# ตั้งค่ากราฟ
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Sound Absorption Coefficient (Alpha)', fontsize=12)
plt.title('Sound Absorption Coefficient vs. Frequency\nfor Different Material Conditions', fontsize=14)
plt.xlim(1, 100)
plt.ylim(0, 0.6)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()

# บันทึกรูป
plt.savefig('absorption_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("สรุปข้อมูล:")
print("=" * 60)
print(f"จำนวนตัวอย่าง: {len(df)} แถว")
print(f"ค่า Proportion1: {df['Proportion1'].unique()}")
print(f"ค่า Proportion2: {df['Proportion2'].unique()}")
print(f"ค่า Temp_C: {df['Temp_C'].unique()}")
print(f"ค่า Pressure_bar: {df['Pressure_bar'].unique()}")
print("=" * 60)
print(f"ค่า Alpha ต่ำสุด: {alpha_values.min().min():.4f}")
print(f"ค่า Alpha สูงสุด: {alpha_values.max().max():.4f}")
print(f"ค่า Alpha เฉลี่ย: {alpha_values.mean().mean():.4f}")
