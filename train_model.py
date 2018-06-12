import numpy as np
from model import Model
from load_data import LoadCifarData

WIDTH = 32
HEIGHT = 32
LR = 1e-3
EPOCHS = 1
MODEL_NAME = 'cifar-model-{}-{}-{}-epochs.model'.format(LR, 'v1', EPOCHS)

model = Model(LR)

(X_train, Y_train), (X_test, Y_test) = LoadCifarData(filepath='cifar-10-python.tar.gz',
                                                            extract_dir='cifar-10-batches-py/', 
                                                            one_hot=True)
##Y_train = to_categorical(Y_train)
##Y_test = to_categorical(Y_test)
##X_train = np.array([i[0] for i in X_train]).reshape(-1,WIDTH,HEIGHT,3)
##Y_train = [i[1] for i in Y_train]
##
##X_test = np.array([i[0] for i in X_test]).reshape(-1,WIDTH,HEIGHT,3)
##Y_test = [i[1] for i in Y_test]


model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=EPOCHS, validation_set=({'input': X_test}, {'targets': Y_test}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

# tensorboard --logdir=foo:F:\play_gta_sa\log

model.save(MODEL_NAME)
