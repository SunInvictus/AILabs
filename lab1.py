import pandas as pd
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X, y = mglearn.datasets.load_extended_boston()
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75,
                                                    random_state=0)

print(f"Форма массива X_train: \n{x_train}")
print(f"Форма массива Y_train:\n{y_train}")
print("*" * 50)
print(f"Форма массива X_test:\n{x_test}")
print(f"Форма массива Y_test:\n{y_test}")
#print("Форма массива X: {}".format(X.shape))