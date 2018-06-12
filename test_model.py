import numpy as np
import cv2
from model import Model

LR = 1e-3
EPOCHS = 15
MODEL_NAME = 'cifar-model-{}-{}-{}-epochs.model'.format(LR, 'v1', EPOCHS)

model = Model(LR)
model.load(MODEL_NAME)

def main():
    img = cv2.imread("dog.jpeg",1)
    resized_img = cv2.resize(img, (32,32))
    resized_img = np.float32(resized_img)
    #screen = cv2.resize(screen,(160,120))
    prediction = model.predict([resized_img.reshape(32,32,3)])[0]
    #prediction = prediction.astype(int)
    prediction = np.argmax(prediction)
    cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(cifar10classes[prediction])

    img = cv2.imread("dog2.jpeg",1)
    resized_img = cv2.resize(img, (32,32))
    resized_img = np.float32(resized_img)
    #screen = cv2.resize(screen,(160,120))
    prediction = model.predict([resized_img.reshape(32,32,3)])[0]
    #prediction = prediction.astype(int)
    prediction = np.argmax(prediction)
    cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(cifar10classes[prediction])

    img = cv2.imread("dog3.jpeg",1)
    resized_img = cv2.resize(img, (32,32))
    resized_img = np.float32(resized_img)
    #screen = cv2.resize(screen,(160,120))
    prediction = model.predict([resized_img.reshape(32,32,3)])[0]
    #prediction = prediction.astype(int)
    prediction = np.argmax(prediction)
    cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(cifar10classes[prediction])
    
    
main()



