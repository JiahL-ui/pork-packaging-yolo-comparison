from pathlib import Path

for split in ["train", "valid", "test"]:
    label_dir = Path("data") / split / "labels"
    if not label_dir.exists():
        print(f"[WARN] {label_dir} not found")
        continue
    for txt in label_dir.glob("*.txt"):
        with open(txt, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"[BAD FORMAT] {txt}:{line_no}: {line}")
                    continue
                cls = int(float(parts[0]))
                if cls >= 4:
                    print(f"[BAD CLASS] {txt}:{line_no}: class={cls} -> {line}")