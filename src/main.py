import os
import pandas as pd
import torch, torchinfo
from tqdm import tqdm

from mask_dataset import MaskedDataset

if __name__ == "__main__":

    model_name = "mbert" 

    dataset = MaskedDataset(model_name=model_name)
    print("DONE")
