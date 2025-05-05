import pickle as pk
from time import perf_counter as pc
import numpy as np  
from random import shuffle
import os


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "."))
    mylist = list(range())
    shuffle(mylist)
    with open("mylist.pkl", "wb") as f:
        start = pc()
        pk.dump(mylist, f)
        end = pc()
        print(f"Pickle dump time: {end - start:.4f} seconds")