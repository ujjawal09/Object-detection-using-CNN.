import urllib
import os
import numpy as np
import pickle
from tflearn.data_utils import to_categorical

def LoadBatch(fpath):
    with open(fpath, 'rb') as f:
      d = pickle.load(f, encoding='latin1')
    data = d["data"]
    labels = d["labels"]
    return data, labels


def LoadCifarData(filepath='cifar-10-python.tar.gz',
                  extract_dir='cifar-10-batches-py/', 
                  one_hot=False):
  X_train = []
  Y_train = []
  for i in range(1, 6):
    fpath = os.path.join(extract_dir, 'data_batch_' + str(i))
    data, labels = LoadBatch(fpath)
    if i == 1:
      X_train = data
      Y_train = labels
    else:
      X_train = np.concatenate([X_train, data], axis=0)
      Y_train = np.concatenate([Y_train, labels], axis=0)

  fpath = os.path.join(extract_dir, 'test_batch')
  X_test, Y_test = LoadBatch(fpath)

  X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                       X_train[:, 2048:])) / 255.
  X_train = np.reshape(X_train, [-1, 32, 32, 3])
  X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                      X_test[:, 2048:])) / 255.
  X_test = np.reshape(X_test, [-1, 32, 32, 3])

  if one_hot:
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

  return (X_train, Y_train), (X_test, Y_test)



if __name__ == '__main__':
  main()
