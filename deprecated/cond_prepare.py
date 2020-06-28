import pandas as pd
import sklearn
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

FILE_PATH = "~/data/pflow_mini_preprocessed.csv"
COND_SAVE_PATH = "~/data/pflow_mini_condition.csv"

dt = pd.read_csv(FILE_PATH, header=None)

cond = dt.loc[:, [6, 7, 9]]
enc = OrdinalEncoder(dtype=int)
cond = enc.fit_transform(cond)

cond = pd.DataFrame(cond)
cond.to_csv(COND_SAVE_PATH, header=None, index=False)
