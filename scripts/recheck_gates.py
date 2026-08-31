import json, glob, os
from finabot.eval.graders import run_hard_gates

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p

for task in ["t007", "t008", "t009", "t016"]:
    d = json.load(open(latest[task], encoding="utf-8"))
    t = d["trials"][0]
    new_gates = run_hard_gates(t["final_text"], {"as_of": d["as_of"]})
    old = t["failed_gates"]
    print(f"{task}: 旧={old or 'ok'}  新(修复后)={new_gates or 'ok'}")
