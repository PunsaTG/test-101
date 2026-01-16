import numpy as np
import pandas as pd
from scipy.stats import qmc

n_samples = 27
n_variables = 3

sampler = qmc.LatinHypercube(d=n_variables, seed=42)
design = sampler.random(n=n_samples)

ranges = {
    'Proportion1': (10, 90),
    'Temp_C': (120, 130),
    'Pressure_bar': (4, 5),
}

data = pd.DataFrame()
for i, (name, (min_v, max_v)) in enumerate(ranges.items()):
    data[name] = min_v + design[:, i] * (max_v - min_v)

data['Proportion1'] = data['Proportion1'].round(0).astype(int)
data['Proportion2'] = 100 - data['Proportion1']
data['Temp_C'] = data['Temp_C'].round(1)
data['Pressure_bar'] = data['Pressure_bar'].round(2)

data = data[['Proportion1', 'Proportion2', 'Temp_C', 'Pressure_bar']]

np.random.seed(42)
for hz in range(1, 1001):
    data[f'Alpha_{hz}Hz'] = np.random.uniform(0.1, 1.0, n_samples).round(4)

print(f"จำนวนแถว: {len(data)}")
print(f"จำนวนคอลัมน์: {len(data.columns)}")
print(data[['Proportion1', 'Proportion2', 'Temp_C', 'Pressure_bar', 'Alpha_1Hz', 'Alpha_2Hz', 'Alpha_3Hz']].head())

data.to_csv('experiment_design_27points.csv', index=False)
print("บันทึกไฟล์: experiment_design_27points.csv")