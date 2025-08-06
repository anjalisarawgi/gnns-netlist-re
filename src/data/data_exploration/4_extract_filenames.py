import os

directory = "../../data/osu035"  
output_path = "data/fileNames_txt/osu035_filenames.txt"

file_list = [f for f in os.listdir(directory) if f.endswith('.gml')]

with open(output_path, 'w') as f:
    for filename in file_list:
        f.write(filename + '\n')