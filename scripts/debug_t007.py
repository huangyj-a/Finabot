import json, glob, os, re
import finabot.eval.graders as g

latest = {}
for p in glob.glob("eval/reports/*/*.json"):
    task = os.path.basename(os.path.dirname(p)).split("-")[0]
    latest[task] = p

d = json.load(open(latest["t007"], encoding="utf-8"))
text = d["trials"][0]["final_text"]
print("=== 新模式下仍命中的匹配 ===")
for pattern in g._UNSUPPORTED_ADVICE_PATTERNS:
    for m in re.finditer(pattern, text):
        start = max(0, m.start() - 20)
        neg = any(neg in text[start:m.start()] for neg in g._NEGATION_WORDS)
        print(f"  pat={pattern!r}")
        print(f"  match={m.group()!r}  neg={neg}")
        print(f"  ctx=...{text[start:m.end()]}...")
        print()
