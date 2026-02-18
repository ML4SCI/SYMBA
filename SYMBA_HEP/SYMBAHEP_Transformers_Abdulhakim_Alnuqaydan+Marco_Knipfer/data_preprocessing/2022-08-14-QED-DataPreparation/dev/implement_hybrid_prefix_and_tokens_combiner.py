# %%
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
import multiprocessing.queues as mpq
import functools
import dill
from typing import Tuple, Callable, Dict, Optional, Iterable, List
  
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from source.read_amplitudes import read_amplitudes, read_amplitudes_and_raw_squares, fix_operator_num_args, get_tree, fix_tree, fix_subscript, fix_subscripts, read_amplitudes_and_squares
import sympy as sp
from source.SympyPrefix import prefix_to_sympy, sympy_to_prefix, simplify_and_prefix, simplify_sqampl
from source.ExpressionsTokensCombiner import combine_m_s, combine_m, shorten_expression

# -------------------------------------------------------------------------------------------  

class TimeoutError(Exception):

    def __init__(self, func, timeout):
        self.t = timeout
        self.fname = func.__name__

    def __str__(self):
            return f"function '{self.fname}' timed out after {self.t}s"


def _lemmiwinks(func: Callable, args: Tuple[object], kwargs: Dict[str, object], q: mp.Queue):
    """lemmiwinks crawls into the unknown"""
    q.put(dill.loads(func)(*args, **kwargs))


def killer_call(func: Callable = None, timeout: int = 10*60) -> Callable:
    """
    Single function call with a timeout

    Args:
        func: the function
        timeout: The timeout in seconds
    """

    if not isinstance(timeout, int):
        raise ValueError(f'timeout needs to be an int. Got: {timeout}')

    if func is None:
        return functools.partial(killer_call, timeout=timeout)

    @functools.wraps(killer_call)
    def _inners(*args, **kwargs) -> object:
        q_worker = mp.Queue()
        proc = mp.Process(target=_lemmiwinks, args=(dill.dumps(func), args, kwargs, q_worker))
        proc.start()
        try:
            return q_worker.get(timeout=timeout)
        except mpq.Empty:
            raise TimeoutError(func, timeout)
        finally:
            try:
                proc.terminate()
            except:
                pass
    return _inners


def _queue_mgr(func_str: str, q_in: mp.Queue, q_out: mp.Queue, timeout: int, pid: int, timeout_logfile: str) -> object:
    """
    Controls the main workflow of cancelling the function calls that take too long
    in the parallel map

    Args:
        func_str: The function, converted into a string via dill (more stable than pickle)
        q_in: The input queue
        q_out: The output queue
        timeout: The timeout in seconds
        pid: process id
    """
    while not q_in.empty():
        positioning, x  = q_in.get()
        q_worker = mp.Queue()
        proc = mp.Process(target=_lemmiwinks, args=(func_str, (x,), {}, q_worker,))
        proc.start()
        try:
            res = q_worker.get(timeout=timeout)
            q_out.put((positioning, res))
        except mpq.Empty:
            q_out.put((positioning, sp.sympify(x)))
            with open(timeout_logfile, "a") as f:
                f.write("Timed out after "+str(timeout)+" seconds. Argument:" + x + "\n")
        finally:
            try:
                proc.terminate()
            except:
                pass


def killer_pmap(func: Callable, iterable: Iterable, cpus: Optional[int] = None, timeout: int = 10*60,
        timeout_logfile = "log/timeout_log.log"):
    """
    Parallelisation of func across the iterable with a timeout at each evaluation

    Args:
        func: The function
        iterable: The iterable to map func over
        cpus: The number of cpus to use. Default is the use max - 2.
        timeout: kills the func calls if they take longer than this in seconds
    """

    if cpus is None:
        cpus = max(mp.cpu_count() - 2, 1)
        if cpus == 1:
            raise ValueError('Not enough CPUs to parallelise. You only have 1 CPU!')
        else:
            print(f'Optimising for {cpus} processors')

    q_in = mp.Queue()
    q_out = mp.Queue()
    sent = [q_in.put((i, x)) for i, x in enumerate(iterable)]

    processes = [
        mp.Process(target=_queue_mgr, args=(dill.dumps(func), q_in, q_out, timeout, pid, timeout_logfile))
        for pid in range(cpus)
    ]

    for proc in processes:
        proc.start()

    result = [q_out.get() for _ in sent]

    for proc in processes:
        proc.terminate()

    return [x for _, x, in sorted(result)]


# -------------------------------------------------------------------------------------------  

# # %%
ampl_folders_prefix = "../2022-08-09-QED_AllParticles_Loop/out/ampl/"
# sqampl_folders_prefix = "../QED_AllParticles_IO/out/sq_ampl/"
sqampl_raw_folders_prefix = "../2022-08-09-QED_AllParticles_Loop/out/sq_ampl_raw/"
# amplitudes_folders_names = ["1to2/", "2to1/", "2to2/", "2to3/", "3to2/", "3to3/",]
# amplitudes_folders_names = ["1to2/", "2to1/", "2to2/"]#, "2to3/", ]# "3to2/", "3to3/",]
amplitudes_folders_names = ["3to2/"]#, "2to3/", ]# "3to2/", "3to3/",]
amplitudes_folders = [ampl_folders_prefix+a for a in amplitudes_folders_names]
# sqamplitudes_raw_folders_names = ["1to2/", "2to1/", "2to2/", "2to3/", "3to2/", "3to3/",]
# sqamplitudes_raw_folders_names = ["1to2/", "2to1/", "2to2/"] #, "2to3/", ]# "3to2/", "3to3/",]
# sqamplitudes_raw_folders_names = ["3to2/"] #, "2to3/", ]# "3to2/", "3to3/",]
sqamplitudes_raw_folders_names = amplitudes_folders_names
sqamplitudes_folders = [sqampl_raw_folders_prefix+a for a in sqamplitudes_raw_folders_names]
cpus = 19
timeout_s = 60*10   # timeout in seconds
# timeout_s = 0.1   # timeout in seconds

progress_file = "log/progress_3to2.log"
outfile_amplitudes =  "../data.nosync/QED_amplitudes_TreeLevel_3to2.txt"
outfile_sqamplitudes_simplified =  "../data.nosync/QED_sqamplitudes_TreeLevel_3to2_simplified.txt"
outfile_sqamplitudes_simplified_prefix =  "../data.nosync/QED_sqamplitudes_TreeLevel_3to2_simplified_prefix.txt"
timeout_logfile = "log/timeout_log_3to2.log"
start_fresh = False   # overwrite progress_file
batch_size = 3000
batch_resume = 0

amplitudes = dict()
sqamplitudes = dict()
for amplitudes_folder, sqamplitudes_folder, name in zip(amplitudes_folders, sqamplitudes_folders, amplitudes_folders_names):
    ic(name)

    amplitudes_files = os.listdir(amplitudes_folder)
    sqamplitudes_files = os.listdir(sqamplitudes_folder)
    ampl, sqampl_raw = read_amplitudes_and_raw_squares(amplitudes_folder, sqamplitudes_folder)

    ampls_prefix = []
    print("Loading amplitudes")
    ctr = 0
    for exp in tqdm(ampl):
        tree = get_tree(exp)
        tree = fix_tree(tree)
        final_expr = fix_subscripts(tree)
        ampls_prefix.append(final_expr)
        ctr = ctr+1
        if ctr>100:
            break

    sqampls_prefix = []
    print("Loading squared amplitudes")
    ctr = 0
    for exp in tqdm(sqampl_raw):
        # simplified = sp.factor(exp)   # worked best for simplification
        # prefix = sympy_to_prefix(simplified)
        # sqampls_prefix.append(prefix)
        sqampls_prefix.append(exp)
        ctr = ctr+1
        if ctr>100:
            break

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

print("simplifying")
cpus=17
timeout_s = 60*1
all_sqamplitudes_simpl = killer_pmap(simplify_sqampl, all_sqamplitudes, cpus=cpus, timeout=timeout_s)

# all_sqamplitudes = [str(x) for x in all_sqamplitudes]
# all_sqamplitudes_simpl = [simplify_sqampl(x) for x in all_sqamplitudes]

all_sqamplitudes_shortened = killer_pmap(shorten_expression, all_sqamplitudes_simpl, cpus=cpus, timeout=timeout_s)



