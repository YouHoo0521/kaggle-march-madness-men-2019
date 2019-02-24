from tabulate import tabulate
import os
from os.path import relpath
from matplotlib import pyplot as plt
from pathlib import Path


def get_project_root() -> Path:
    '''Returns project root folder'''
    return Path(__file__).parent.parent


# Notebook base directory
NOTEBOOK_BASEDIR = os.path.join(get_project_root(), 'notebooks')
NOTEBOOK_IMGDIR = os.path.join(NOTEBOOK_BASEDIR, 'figs')


##################################################
# Helper functions for pretty output from org-mode
##################################################

def create_print_df_fcn(tablefmt='orgtbl'):
    def print_df(df, tablefmt=tablefmt):
        print(tabulate(df, headers="keys", tablefmt=tablefmt))
    return print_df


def create_show_fig_fcn(img_dir='tmp/'):
    img_dir = os.path.join(NOTEBOOK_IMGDIR, img_dir)
    os.makedirs(img_dir, exist_ok=True)

    def show_fig(fname, img_dir=img_dir):
        fpath = os.path.join(img_dir, fname)
        plt.savefig(fpath)
        plt.close()
        print(relpath(fpath, os.path.curdir))
    return show_fig
