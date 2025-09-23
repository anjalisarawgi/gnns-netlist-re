# Minimum working Example to create graph with partitioning information

## Requirements
1. netlist in .v (verilog) format, in this example in netlist
2. Partition Information as .csv file, in this example in partition folder
3. Technology Library Information, in this case osu035_stdcells.lib

## First Step: Create Adjlist from verilog netlist for each netlist in netlist folder

1. make new folder adjlist
2. ```verilog2graph netlist --output_format txt  --do-not-zip -o adjlist/ -l osu035_stdcells.lib```

## Second Step: Create .pq graph files adjlists using the partition files

Note: this is necessary for the third step, so that we have a graph file for each partition

1. ```run_clustering adjlist/ -i partion/aes_cipher/ -u osu035 --task partition```
2. output should be saved in newly created folder "outputFiles"

## Third Step: create graph (gml) with partition label for "top" adjlist, in this case only for the "top" file, or really any file that has subcomponents

1. ```partition2gephi adjlist/aes_cipher_top.txt --do-not-zip -p outputFiles/osu035/partition_graph/ -t gml -o . ```
2. output is saved in current folder, this can be e.g. opened in gephi, or inspected manually

### verilog2graph netlist --output_format txt  --do-not-zip -o adjlist/ -l osu035_stdcells.lib
### run_clustering adjlist/ -i partition/aes_cipher -u osu035 --task partition 
### partition2gephi adjlist/aes_cipher_top.txt --do-not-zip -t gml -o . -p outputFiles/osu035/partition_graph/ --cores 1


### verilog2graph tum-eisec-benchmarks-main/netlist/des_latest/verilog/osu035/des.v --output_format txt  --do-not-zip -o adjlist/des_latest/osu035 -l lib/osu035_stdcells.lib
### run_clustering adjlist/des_latest/osu035 -i tum-eisec-benchmarks-main/partition/des_latest/osu035/ -u des_latest/osu035/ --task partition --cores 1
### partition2gephi adjlist/des_latest/osu035/des.txt --do-not-zip -t gml -o ../graphs/raw/des_latest/osu035 -p outputFiles/des_latest/osu035/partition_graph --cores 1