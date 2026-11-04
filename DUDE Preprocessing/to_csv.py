import glob, json, os
import pandas as pd

root = r"data\dude_processed"
for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
    df = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    for c in df.columns:
        df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False)
                          if isinstance(v, (list, dict)) else v)
    out = path[:-len(".jsonl")] + ".csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(out, len(df), "rows")