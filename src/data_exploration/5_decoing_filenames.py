import csv
import pandas as pd

input_file = "data/fileNames_txt/all_filenames.txt"
output_file = "data/fileNames_txt/decoded.csv"

# we do this to check the file names and decode them just for understanding
with open(input_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

data = []
for line in lines:
    parts = line.split('_')
    part_a = '_'.join(parts[:2])  # "automated_gscl45nm"

    try:
        final_index = parts.index('final')
        part_b = parts[final_index - 1] if final_index > 0 else ''
        part_c = '_'.join(parts[final_index + 1:]).replace('.gml', '')
    except ValueError:
        part_b = ''
        part_c = ''

    data.append((line, part_a, part_b, part_c))


with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'part_a', 'part_b', 'part_c'])
    writer.writerows(data)

print(f"CSV saved to {output_file}")

df = pd.read_csv(output_file)
df['part_a'].drop_duplicates().to_csv("data/fileNames_txt/unique_part_a.csv", index=False, header=True)
df['part_b'].drop_duplicates().to_csv("data/fileNames_txt/unique_part_b.csv", index=False, header=True)
df['part_c'].drop_duplicates().to_csv("data/fileNames_txt/unique_part_c.csv", index=False, header=True)
