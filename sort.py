input_file = "all_designs.txt"      # your original txt
output_file = "all_designs_sorted.txt"  # sorted output

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

lines.sort()

with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")