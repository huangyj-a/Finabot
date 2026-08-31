import json, glob, os

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p

for task in ["t019"]:
    if task not in latest:
        print(task, "NO REPORT"); continue
    d = json.load(open(latest[task], encoding="utf-8"))
    t = d["trials"][0]
    print(f"=== {task} final_text repr: {t['final_text'][:200]!r}")
    print(f"=== trace reports ===")
    reports = t["trace"].get("reports", {})
    for k, v in reports.items():
        print(f"  {k}: {str(v)[:80]!r}")
    print(f"=== trace messages types ===")
    for m in t["trace"].get("messages", []):
        print(f"  {m['type']}: {str(m['content'])[:60]!r}")
