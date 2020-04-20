#import tensorflow as tf 
import cv2
import numpy as np
from threading import Thread
import time
from keras.models import load_model
REV_CLASS_MAP = {
    0: "p",
    1: "f",
    2: "c",
    3: "none"
}

model = load_model('models/rock.h5')
def image2move(img):
    pred = model.predict(np.array([img],dtype=np.float32))
    move_code = np.argmax(pred)

    return REV_CLASS_MAP[move_code]

