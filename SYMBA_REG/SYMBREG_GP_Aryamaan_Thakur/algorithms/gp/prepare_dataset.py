import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser("Prepare AI Feynman Dataset", add_help=False)
    parser.add_argument('--dataset_dir', default="../Feynman_with_units", type=str)
    parser.add_argument('--dataframe_path', default="FeynmanEquations.csv", type=str)
    parser.add_argument('--chunk_size', default=200, type=int)
    parser.add_argument('--output_dir', default="dataset", type=str)

    return parser

#def progressBar(iterable, decimals = 1, length = 100, fill = '█', printEnd = "\r"):
#    """
#    Call in a loop to create terminal progress bar
#    @params:
#        iterable    - Required  : iterable object (Iterable)
#        prefix      - Optional  : prefix string (Str)
#        suffix      - Optional  : suffix string (Str)
#        decimals    - Optional  : positive number of decimals in percent complete (Int)
#        length      - Optional  : character length of bar (Int)
#        fill        - Optional  : bar fill character (Str)
#        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
#    """
#    total = len(list(iterable))
#    # Progress Bar Printing Function
#    def printProgressBar (iteration):
#        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#        filledLength = int(length * iteration // total)
#        bar = fill * filledLength + '-' * (length - filledLength)
#    # Initial Call
#    printProgressBar(0)
#    # Update Progress Bar
#    for i, item in enumerate(iterable):
#        yield item
#        printProgressBar(i + 1)
#    # Print New Line on Complete


def main(args):
    df = pd.read_csv(args.dataframe_path)
    
    train_df = {
        "filename":[],
        "data_num":[],
        "number":[]
        }

    for index, row in tqdm(df.iterrows()):
#    for index, row in progressBar(df.iterrows(), length = 50):
        path = os.path.join(args.dataset_dir, row["Filename"])
        with open(path) as file:
            data = file.readlines()
        data = np.array([i.split() for i in data], dtype=np.float32)
        n_splits = data.shape[0] // args.chunk_size
        data = data[:n_splits*args.chunk_size]
        chunks = np.split(data, n_splits)

        sub_dir = os.path.join(args.output_dir, row["Filename"])
        os.makedirs(sub_dir, exist_ok=True)

        for (index, chunk) in enumerate(chunks):
            np.save(os.path.join(sub_dir, f"{index}.npy"), chunk)

        train_df["filename"].extend([row["Filename"]]*n_splits)
        train_df["data_num"].extend([i for i in range(n_splits)])
        train_df["number"].extend([row["Number"] for i in range(n_splits)])

    train_df = pd.DataFrame(train_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare AI Feynman Dataset', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    main(args)
