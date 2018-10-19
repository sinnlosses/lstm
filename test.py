
import csv

fname = "./templete_model/models_5000_fast_isTraining/sample_result_weights.hdf5_0_1.csv"
save_fname = "temp.txt"
with open(fname, "r") as fi:
    with open(save_fname, "w") as fo:
        reader = csv.reader(fi)
        for line in reader:
            fo.write("-----\n")
            fo.write(line[0]+"\n")








