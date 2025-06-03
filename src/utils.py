import random
from math import *
from collections import Counter
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings

random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

