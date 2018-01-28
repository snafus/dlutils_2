import math,os,json,sys,re

import pickle
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
import scipy

import bcolz



import keras
from keras import backend as K
from keras.utils.np_utils import to_categorical


# shuffles
def shuffle(p):
    """Randomise the order of an input sequence.
    Does not change input.
    """
    return np.random.permutation(p)


# data manipulation
def join_columns(list_of_np_arrays,axis=1):
    """Join two columns together as new output"""
    return np.stack(list_of_np_arrays, axis=axis)
    

# data persistence
def save_array(fname, arr):
    """Use bcolz to save numpy array"""
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    """Use bcolz to load numpy array"""    
    return bcolz.open(fname)[:]


# data transformations
def to_onehot(arr):
    """Turn array of integers into matrix of binary [1,0] encodeed values.
    """
    return to_categorical(arr)


def add_dimension(array,dimension_index=1):
    """Insert a new axis that will appear at the `dimension_index` position in the expanded
array shape.
    For example, in [nsamples, x,y] image; to convert to
    [nsamples,1,x,y] with channel number as additional dimension.
    Use channel_index for position of the new dimension.
    """
    return np.expand_dims(array,channel_index)

