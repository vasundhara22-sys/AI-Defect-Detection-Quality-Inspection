import pandas as pd

orig = pd.read_csv("reports/batch_neu.csv").set_index("file")["status"]
rob = pd.read_csv("reports/batch_robust.csv")
rob["variant"] = rob["file"].str.extract(r"-([a-z0-9]+)\.jpg$")[0]
rob["orig_file"] = rob["file"].str.replace(r"-[a-z0-9]+\.jpg$", ".jpg", regex=True)
rob["orig_status"] = rob["orig_file"].map(orig)
rob["same_status"] = rob["status"] == rob["orig_status"]

rows = []
for v, g in rob.groupby("variant"):
    rows.append({
        "variant": v,
        "images": len(g),
        "Pass": int((g["status"] == "Pass").sum()),
        "Review": int((g["status"] == "Review Required").sum()),
        "Fail": int((g["status"] == "Fail").sum()),
        "same_as_original_%": round(100 * g["same_status"].mean()),
    })
print(pd.DataFrame(rows).to_string(index=False))
print()
print("Total Pass (missed defects):", int((rob["status"] == "Pass").sum()), "of", len(rob))
rob.to_csv("reports/robust_summary.csv", index=False)