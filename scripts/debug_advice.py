import json, glob, os, re
import finabot.eval.graders as g

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p

for task in ["t007", "t009", "t016"]:
    d = json.load(open(latest[task], encoding="utf-8"))
    text = d["trials"][0]["final_text"]
    print(f"===== {task} =====")
    for pattern in g._UNSUPPORTED_ADVICE_PATTERNS:
        for m in re.finditer(pattern, text):
            start = max(0, m.start() - 16)
            ctx = text[start:m.end()]
            neg = any(neg in text[start:m.start()] for neg in g._NEGATION_WORDS)
            print(f"  match={m.group()!r}  neg_in_window={neg}  ctx=...{ctx}...")
