import json, glob, os

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p  # 后出现的覆盖（glob 按 mtime 升序）

for task in sorted(latest):
    d = json.load(open(latest[task], encoding="utf-8"))
    t = d["trials"][0]
    gates = t["failed_gates"] or "ok"
    print(f"{task}: quality={t['quality']:5.1f}  gates={gates}  severe={t['severe']}")
