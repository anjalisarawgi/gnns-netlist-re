from pathlib import Path
import yaml  # pip install pyyaml

cfg_path = "config/train_336_m2.yml"  # put that whole block in this file

cfg = yaml.safe_load(Path(cfg_path).read_text())
gml_paths = cfg["train_gml"]   # <-- this is now a normal Python list

log_file = "gml_missing_boundary.log"

missing_boundary = []
for p in gml_paths:
    path = Path(p)
    if not path.exists():
        missing_boundary.append(f"{path}  [FILE NOT FOUND]")
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "boundary" not in text:
            missing_boundary.append(str(path))
    except Exception as e:
        missing_boundary.append(f"{path}  [ERROR: {e}]")

Path(log_file).write_text("\n".join(missing_boundary) + ("\n" if missing_boundary else ""))
print(f"Checked {len(gml_paths)} files")
print(f"Files missing 'boundary': {len(missing_boundary)}")
print(f"Log written to: {log_file}")