import pathlib

text = open(r"d:\d\hackathon\sih2026\src\ingest.py", encoding="utf-8").read()
# just verify parse
import ast
ast.parse(text)
print("current file parses ok, lines:", len(text.splitlines()))