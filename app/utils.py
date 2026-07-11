import os
from pathlib import Path
import pandas as pd

def load_data():
    """Safely handles data ingestion across differing execution contexts."""
    # Find base directory context
    current_dir = Path(os.getcwd())
    if current_dir.name == "app":
        project_root = current_dir.parent
    else:
        project_root = current_dir
        
    raw_data_path = project_root / "data" / "raw" / "eep_baseline.csv"
    
    # Visual terminal debugging
    print(f"\n🔍 [DEBUG] Streamlit is looking for data at: {raw_data_path.resolve()}")
    print(f"🔍 [DEBUG] Does file exist? {raw_data_path.exists()}")
    
    if raw_data_path.exists():
        return pd.read_csv(raw_data_path)
    else:
        # Fallback to prevent app from breaking if directory context shifts
        fallback_data = {
            "Fiscal_Year": ["2021/22", "2022/23", "2023/24", "2024/25", "2025/26 (Proj)"],
            "Era": ["Pre-Data Mining", "Pre-Data Mining", "Transition Phase", "Post-Data Mining", "Post-Data Mining"],
            "Total_Revenue_Billion_Birr": [18.5, 22.0, 27.0, 75.4, 109.0],
            "Net_Income_Loss_Billion_Birr": [-26.2, -24.8, -10.0, 5.2, 14.2],
            "Total_FX_Revenue_Million_USD": [112.0, 125.0, 140.0, 338.0, 410.0],
            "Data_Mining_FX_Contribution_Million_USD": [0.0, 0.0, 15.0, 220.0, 285.0],
            "Regional_Export_FX_Contribution_Million_USD": [112.0, 125.0, 125.0, 118.0, 125.0],
            "Total_Grid_Generation_GWh": [15400, 18200, 22100, 29480, 34500],
            "Data_Mining_Consumption_GWh": [0, 0, 1100, 7960, 10350],
            "Data_Mining_Grid_Share_Pct": [0.0, 0.0, 5.0, 27.0, 30.0]
        }
        return pd.DataFrame(fallback_data)
    