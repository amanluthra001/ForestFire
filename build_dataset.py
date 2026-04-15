import pandas as pd
import os

weather_dir = r"C:\Users\bhavy\OneDrive\Desktop\minor project\weather"
burn_dir = r"C:\Users\bhavy\OneDrive\Desktop\minor project\satellite-burn"

data_rows = []

for q in range(1, 25):  # 1 to 24
    weather_file = os.path.join(weather_dir, f"group_{q}.csv")
    burn_file = os.path.join(burn_dir, f"group_{q}.csv")

    # Read weather (36 rows)
    w = pd.read_csv(weather_file)
    w = w.reset_index(drop=True)
    w['cell_id'] = range(1, 37)
    w['quarter'] = q

    # Read burn (36 values)
    b = pd.read_csv(burn_file, header=None)
    b.columns = ['burn_value']
    b['cell_id'] = range(1, 37)

    # Merge w + b
    df = w.merge(b, on="cell_id")
    data_rows.append(df)

final_df = pd.concat(data_rows, ignore_index=True)

# Save final dataset
final_df.to_csv("dataset.csv", index=False)
print("dataset.csv created successfully")
