# %%
# script not working since some `factor` never finish
import sys
import os
from icecream import ic 
import csv
import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import sympy as sp
# from tqdm.notebook import tqdm
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from parallelbar import progress_imap, progress_map

  
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from source.read_amplitudes import read_amplitudes, read_amplitudes_and_raw_squares, fix_operator_num_args, get_tree, fix_tree, fix_subscript, fix_subscripts, read_amplitudes_and_squares
import sympy as sp
from source.SympyPrefix import prefix_to_sympy, sympy_to_prefix, simplify_and_prefix

  

# # %%
ampl_folders_prefix = "../QED_AllParticles_IO/out/ampl/"
# sqampl_folders_prefix = "../QED_AllParticles_IO/out/sq_ampl/"
sqampl_raw_folders_prefix = "../QED_AllParticles_IO/out/sq_ampl_raw/"
amplitudes_folders_names = ["1to2/", "2to1/", "2to2/", "2to3/", "3to2/", "3to3/",]
amplitudes_folders = [ampl_folders_prefix+a for a in amplitudes_folders_names]
sqamplitudes_raw_folders_names = ["1to2/", "2to1/", "2to2/", "2to3/", "3to2/", "3to3/",]
sqamplitudes_folders = [sqampl_raw_folders_prefix+a for a in sqamplitudes_raw_folders_names]
cpus = 20

progress_file = "log/progress_up_to_3to3.log"
outfile_amplitudes =  "../data.nosync/QED_amplitudes_TreeLevel_UpTo3to3.txt"
outfile_sqamplitudes =  "../data.nosync/QED_sqamplitudes_TreeLevel_UpTo3to3_simplified.txt"
start_fresh = True   # overwrite progress_file
batch_size = 1000
batch_resume = 0

#
# # %%
# # # should really parallelize the simplify etc. Here is a link:
# # # https://www.delftstack.com/howto/python/parallel-for-loops-python/
# # # the code will look like this:
#
# # import multiprocessing
#
#
# # def sumall(value):
# #     return sum(range(1, value + 1))
#
# # pool_obj = multiprocessing.Pool()
#
# # answer = pool_obj.map(sumall,range(0,5))
# # print(answer)
#
# # %%
#
amplitudes = dict()
sqamplitudes = dict()
for amplitudes_folder, sqamplitudes_folder, name in zip(amplitudes_folders, sqamplitudes_folders, amplitudes_folders_names):
    ic(name)

    amplitudes_files = os.listdir(amplitudes_folder)
    sqamplitudes_files = os.listdir(sqamplitudes_folder)
    ampl, sqampl_raw = read_amplitudes_and_raw_squares(amplitudes_folder, sqamplitudes_folder)

    ampls_prefix = []
    print("Loading amplitudes")
    for exp in tqdm(ampl):
        tree = get_tree(exp)
        tree = fix_tree(tree)
        final_expr = fix_subscripts(tree)
        ampls_prefix.append(final_expr)

    sqampls_prefix = []
    print("Loading squared amplitudes")
    for exp in tqdm(sqampl_raw):
        # simplified = sp.factor(exp)   # worked best for simplification
        # prefix = sympy_to_prefix(simplified)
        # sqampls_prefix.append(prefix)
        sqampls_prefix.append(exp)
    amplitudes[name] = ampls_prefix
    sqamplitudes[name] = sqampls_prefix

# # %%
all_amplitudes = []
for key in amplitudes.keys():
    for x in amplitudes[key]:
        all_amplitudes.append(x)

all_sqamplitudes = []
for key in sqamplitudes.keys():
    for x in sqamplitudes[key]:
        all_sqamplitudes.append(x)
#
# # %%
ic(len(all_amplitudes))
ic(len(all_sqamplitudes))
#
# # %%
def get_unique_indices(l):
    seen = set([0])
    res = []
    for i, n in enumerate(l):
        if n not in seen:
            res.append(i)
            seen.add(n)
    return res
#
# # %%
all_amplitudes_str = [''.join(a) for a in all_amplitudes]
#
# # %%
unique_indices = np.unique(all_amplitudes_str, return_index=True, axis=0)[1]
unique_indices_sq = np.unique(all_sqamplitudes, return_index=True, axis=0)[1]
#
# # %%
unique_indices.sort()
unique_indices_sq.sort()
#
# # %%
ic(len(unique_indices));
ic(len(unique_indices_sq));
ic(len(unique_indices_sq) / len(unique_indices));
#
# # %%
all_amplitudes_unique = [all_amplitudes[i] for i in unique_indices]
all_sqamplitudes_unique = [all_sqamplitudes[i] for i in unique_indices]
#
# %% [markdown]
# All amplitudes are unique, but only 54% of squared amplitudes are unique.
# I will still keep all of them.
num_batches = len(unique_indices) // batch_size + 1

if start_fresh:
    print("Starting fresh. Deleting "+progress_file)
    if os.path.exists(progress_file):
        os.remove(progress_file)
    if os.path.exists(outfile_amplitudes):
        print("Deleting "+outfile_amplitudes)
        os.remove(outfile_amplitudes)
    if os.path.exists(outfile_sqamplitudes):
        print("Deleting "+outfile_sqamplitudes)
        os.remove(outfile_sqamplitudes)

    with open(progress_file, 'a') as f:
        f.write("Starting calculation at:"+str(datetime.now())+"\n")
        f.write("Worked on folders:"+"".join(amplitudes_folders_names)+"\n")
        f.write("batch_size:"+str(batch_size)+"\n")
        f.write("lines:"+str(len(unique_indices))+"\n")
        f.write("num_batches:"+str(num_batches)+"\n")
        f.write("----------------------\n")
        f.write("----------------------\n")

else:
    print("Resuming calculations, reading progress from "+progress_file)
    with open(progress_file) as f:
        progess_file_contents = [line for line in f.readlines()]
    batch_resume = int(progess_file_contents[-7].split(":")[1]) + 1
    index_resume = int(progess_file_contents[-3].split(":")[1])
    batch_size_resume = int(progess_file_contents[-2].split(":")[1])
    assert batch_resume*batch_size_resume == index_resume
    batch_size = batch_size_resume
    ic(batch_resume)
    ic(index_resume)
    ic(batch_size_resume)
    print("Continuing with batch " +str(batch_resume)+"/"+str(num_batches)+", which amounts to line "+str(index_resume)+".")
    if batch_resume == num_batches:
        print("Nothing to resume, already finished.")

batches = range(batch_resume, num_batches)

print("Simplifying amplitudes in batches of "+str(batch_size)+":")
for batch in tqdm(batches):
    batch_start_index = batch*batch_size
    batch_end_index = (batch+1)*batch_size
    sqamplitudes_batch = all_sqamplitudes_unique[batch_start_index:batch_end_index]
    amplitudes_batch = all_amplitudes_unique[batch_start_index:batch_end_index]
    print("batch:", batch, "/", len(batches))

    start_time = datetime.now()
    # ------------------------------
    with mp.Pool(processes=cpus) as p:
        # sqamplitudes_simplified_prefix_batch = progress_map(simplify_and_prefix, sqamplitudes_batch, n_cpu=cpus)  #, core_progress=True)
        sqamplitudes_simplified_prefix_batch = p.map(simplify_and_prefix, sqamplitudes_batch)
    ## output not working yet!
    ## try to append to file in batches and not each line
    out_amplitudes_str = [",".join(x) for x in amplitudes_batch]
    out_amplitudes_str = "\n".join(out_amplitudes_str)+"\n"
    out_sqamplitudes_str = [",".join(x) for x in sqamplitudes_simplified_prefix_batch]
    out_sqamplitudes_str = "\n".join(out_sqamplitudes_str)+"\n"

    with open(outfile_amplitudes, "a") as f:
        f.write(out_amplitudes_str)
    with open(outfile_sqamplitudes, "a") as f:
        f.write(out_sqamplitudes_str)

    # ------------------------------
    end_time = datetime.now()

    with open(progress_file, 'a') as f:
        f.write("Appended amplitudes to "+outfile_amplitudes+"\n")
        f.write("Appended sqamplitudes to "+outfile_sqamplitudes+"\n")
        f.write("batch:" + str(batch) + "\n")
        f.write("started at:" + str(start_time) + "\n")
        f.write("finished at:" + str(end_time) + "\n")
        f.write("batch_start_index:" + str(batch_start_index) + "\n")
        f.write("batch_end_index:" + str(batch_end_index) + "\n")
        f.write("batch_size:" + str(batch_size) + "\n")
        f.write("----------------------\n")


# # %%
# X = []
# with open(outfile_amplitudes, 'r') as f:
#     for line in f.readlines() :
#         line = line.split(";")
#         # have to remove new line character for some reason
#         line[-1] = line[-1].replace("\n", "")
#         X.append(line)
#
# y = []
# with open(outfile_sqamplitudes, 'r') as f:
#     for line in f.readlines() :
#         line = line.split(";")
#         # have to remove new line character for some reason
#         line[-1] = line[-1].replace("\n", "")
#         y.append(line)
#
# # %%
# ic(X == all_amplitudes_unique);
# ic(y == all_sqamplitudes_simplified_prefix);
#
# # %%
# ampls_lens = [len(x) for x in X]
# sqampls_lens = [len(x) for x in y]
#
# # %%
# plt.hist(ampls_lens, label="Amplitudes lengths")
# plt.hist(sqampls_lens, label="Squared amplitudes lengths")
# plt.legend()
# ic(np.mean(ampls_lens))
# ic(np.mean(sqampls_lens))
#
# # %%
#
#
#
