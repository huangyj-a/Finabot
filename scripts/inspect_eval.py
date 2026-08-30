import json, glob, os

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p

for task in ["t017", "t019", "t001", "t008"]:
    if task not in latest:
        print(task, "NO REPORT")
        continue
    d = json.load(open(latest[task], encoding="utf-8"))
    t = d["trials"][0]
    print(f"===== {task} quality={t['quality']} gates={t['failed_gates']} =====")
    print(t["final_text"][:300])
    print()
