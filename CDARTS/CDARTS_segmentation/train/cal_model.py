from __future__ import division
import os
import sys
import time
import glob
import json
import logging
import argparse
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main():

    # Import Data
    model_files = glob.glob("train_samples/*.json")

    
    for model_file in model_files:

        with open(model_file, 'r') as f:
            # dict_a = json.loads(f, cls=NpEncoder)
            model_dict = json.loads(f.read())


    df = pd.read_csv("mpg_ggplot2.csv")

    # Prepare data
    x_var = 'manufacturer'
    groupby_var = 'class'
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    # Draw
    plt.figure(figsize=(16,9), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 40)
    plt.xticks(rotation=90, horizontalalignment='left')
    plt.show()
    




if __name__ == '__main__':
    main()

