import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from load_data import LoadCifarData

(X, Y), (X_test, Y_test) = LoadCifarData()
X, Y = shuffle(X, Y)
Y = to_categorical(Y)
Y_test = to_categorical(Y_test)

process = ImagePreprocessing()
augment = ImageAugmentation
process.add_featurewise_zero_center()
process.add_featurewise_stdnorm()
augment.add_random_flip_leftright()
augment.add_random(rotation(max_angle=25.))
                   
# Convolutional network building
def Model(lr):
	network = input_data(shape=[None, 32, 32, 3],
                             data_preprocessing=process,
                             data_augmentation=augment,
	                     name='input')
	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 10, activation='softmax')
	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=lr, name='targets')

	# Train using classifier
	model = tflearn.DNN(network, checkpoint_path='model', max_checkpoints=1,tensorboard_verbose=0, tensorboard_dir='log')

	return model


