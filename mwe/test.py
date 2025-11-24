from pathlib import Path

designs = [
    "aes_decrypt_fpga_latest",
    "ecg_latest_final_func11",
    "sha3_latest",
    "sha_core_latest",
    "xtea_latest",

    "present_encryptor_latest",
    "gost28147-89_latest",
    "hight_latest",
]

base = Path("tum-eisec-benchmarks-main")

for d in designs:
    part_root = base / "partition" / d
    print(f"\n=== {d} ===")

    if not part_root.exists():
        print("  [MISSING] partition/<design>/ directory")
        continue

    techs = [p for p in part_root.iterdir() if p.is_dir()]
    print("  tech dirs:", [t.name for t in techs])

    for t in techs:
        csv_files = list(t.glob("*.csv"))
        print(f"  {t}: found {len(csv_files)} hierarchy CSV files")

        # Preview first 5
        for f in csv_files[:5]:
            print("     -", f.name)

        if len(csv_files) == 0:
            print("     No CSV partitions available")