# import all necessary libraries
import os
import pandas as pd

import numpy as np
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm_notebook as tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import wordcloud
from wordcloud import WordCloud, STOPWORDS


# announce all necessary dirs
def get_dirs(is_kaggle=True):
    if is_kaggle:
        KAGGLE_INPUT = '/kaggle/input'
        MOVIELENS_DIR = os.path.join(KAGGLE_INPUT, 'movielens-20m-dataset')
    else:
        WORK_DIR = '~/Movielens'
        MOVIELENS_DIR = os.path.join(WORK_DIR, 'movielens-20m-dataset')
    RATINGS_DIR = os.path.join(MOVIELENS_DIR, 'rating.csv')
    MOVIES_DIR = os.path.join(MOVIELENS_DIR, 'movie.csv')
    TAGS_DIR = os.path.join(MOVIELENS_DIR, 'tag.csv')
    LINKS_DIR = os.path.join(MOVIELENS_DIR, 'link.csv')
    GENOME_TAGS_DIR = os.path.join(MOVIELENS_DIR, 'genome_tags.csv')
    GENOME_SCORES_DIR = os.path.join(MOVIELENS_DIR, 'genome_scores.csv')

    return {
        'rating_path': RATINGS_DIR,
        'movie_path': MOVIES_DIR,
        'tag_path': TAGS_DIR,
        'link_path': LINKS_DIR,
        'genome_tags_path': GENOME_TAGS_DIR,
        'genome_scores_path': GENOME_SCORES_DIR
    }


def evaluate(predictions, item_based_matrix):
    """
    calculate RMSE between known values and predicted
    """
    from sklearn.metrics import mean_squared_error

    mask = item_based_matrix > 0
    ground_truth = item_based_matrix[mask].flatten()
    actual = predictions[mask].flatten()

    return mean_squared_error(actual, ground_truth, squared=False)