import re

with open("frontend/app.py", "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(
    r'm2\.metric\("CANDIDATE.*?", n_cand\)',
    'n_watch = sum(1 for c in clusters if c["status"] == "WATCHLIST")\\n    m2.metric("WATCHLIST / CANDIDATE", n_cand + n_watch)',
    content
)

content = content.replace("forensic visual inspector", "visual triage assistant")

with open("frontend/app.py", "w", encoding="utf-8") as f:
    f.write(content)
