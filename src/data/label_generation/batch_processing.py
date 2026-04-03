import json5
from pathlib import Path
from shared.data_processing_shared import process_verilog

def main():
    with open("shared/verilog_files.jsonc") as f:
        config = json5.load(f)

    verilog_files = config.get("verilog_files", [])
    if not verilog_files:
        print("No verilog_files found in config.json")
        return

    for file_path in verilog_files:
        p = Path(file_path)
        if not p.exists():
            print(f"[WARNING] Skipping missing file: {p}")
            continue

        print(f"\n=== Processing {p} ===")
        process_verilog(p)

if __name__ == "__main__":
    main()