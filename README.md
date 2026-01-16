# Material Properties Optimization Project

โปรเจคนี้เป็นการศึกษาและพัฒนาโมเดล Machine Learning สำหรับทำนายและหาค่าพารามิเตอร์ที่เหมาะสมที่สุดของวัสดุ

## 📋 Project Overview

โปรเจคนี้ใช้ XGBoost และ Gaussian Process Regression (GPR) ในการ:
- ทำนายค่า Alpha ของวัสดุที่ความถี่ต่างๆ (1Hz - 100Hz)
- หาค่า Input ที่เหมาะสมที่สุด (Proportion1, Proportion2, Temp_C, Pressure_bar) เพื่อให้ได้ค่า Alpha ตามเป้าหมาย

## 🗂️ Project Structure

```
├── README.md               <- Project description and setup guide
├── data
│   ├── external            <- Data from third party sources
│   ├── interim             <- Intermediate data that has been transformed
│   ├── processed           <- The final, canonical data sets for modeling
│   └── raw                 <- The original, immutable data dump
│
├── docs                    <- Project documentation
│
├── models                  <- Trained and serialized models, model summaries
│
├── notebooks               <- Jupyter notebooks (Naming: 1.0-exploration.ipynb)
│
├── references              <- Data dictionaries, manuals, and explanatory materials
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Graphics and plots for reports
│
├── requirements.txt        <- The dependencies file for reproducing the environment
│
├── src                     <- Source code for use in this project
│   ├── __init__.py         <- Makes src a Python module
│   ├── data                <- Scripts to download or generate data
│   ├── features            <- Scripts to turn raw data into features for modeling
│   ├── models              <- Scripts to train models and then use trained models
│   └── visualization       <- Scripts to create exploratory and results oriented visualizations
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PunsaTG/test-101.git
cd test-101
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data

- **material_100hz_data.csv** - ข้อมูลวัสดุที่ความถี่ 100Hz
- **experiment_design_27points.csv** - ข้อมูลการออกแบบการทดลอง 27 จุด

## 🧪 Usage

### Training Models
ดู notebooks สำหรับการ train โมเดล:
- `XGboost.ipynb` - โมเดล XGBoost พื้นฐาน
- `XGboost+GPR_Fast.ipynb` - XGBoost + GPR เวอร์ชันเร็ว
- `CheckTheBestModel.ipynb` - ตรวจสอบโมเดลที่ดีที่สุด

### Finding Optimal Parameters
```bash
python find_optimal_input.py
```

## 📈 Results

ผลลัพธ์การวิเคราะห์จะถูกบันทึกใน:
- `reports/figures/` - กราฟและ plots
- `models/` - โมเดลที่ train แล้ว

## 📝 License

This project is licensed under the MIT License.

## ✉️ Contact

PunsaTG - GitHub: [PunsaTG](https://github.com/PunsaTG)
