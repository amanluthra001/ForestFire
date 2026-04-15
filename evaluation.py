import os
import pandas as pd

PRED_ROOT = "results_opt\predictions"
WEATHER_ROOT = "weather"
OUT_DIR = "evaluation"

os.makedirs(OUT_DIR, exist_ok=True)

all_regions = {}
false_negatives = []

for q_folder in sorted(os.listdir(PRED_ROOT)):
    if not q_folder.startswith("quarter_"):
        continue

    quarter = int(q_folder.split("_")[1])
    q_path = os.path.join(PRED_ROOT, q_folder)

    weather_path = os.path.join(WEATHER_ROOT, f"group_{quarter}.csv")
    weather_df = pd.read_csv(weather_path)

    weather_df["region"] = range(1, len(weather_df) + 1)
    weather_df = weather_df.set_index("region")

    for file in sorted(os.listdir(q_path)):
        if not file.endswith(".csv"):
            continue

        fpath = os.path.join(q_path, file)
        df_pred = pd.read_csv(fpath)

        region = int(df_pred["region"].iloc[0])
        actual = int(df_pred["actual"].iloc[0])
        pred = int(df_pred["pred_label"].iloc[0])

        prob_value = float(df_pred["pred_prob"].iloc[0])

        if pred == 1 and actual > 0:
            outcome = "TP"
        elif pred == 1 and actual == 0:
            outcome = "FP"
        elif pred == 0 and actual > 0:
            outcome = "FN"
            false_negatives.append((region, quarter))
        else:
            outcome = "TN"

        weather_row = weather_df.loc[region].to_dict() if region in weather_df.index else {}

        row = {
            "quarter": quarter,
            "actual": actual,
            "pred": pred,
            "pred_prob": prob_value,
            "outcome": outcome,
        }
        row.update(weather_row)

        if region not in all_regions:
            all_regions[region] = []

        all_regions[region].append(row)

for region, rows in all_regions.items():
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("quarter")
    df_out.to_csv(os.path.join(OUT_DIR, f"region_{region}.csv"), index=False)

print("\nAll evaluation CSVs generated in:", OUT_DIR)
