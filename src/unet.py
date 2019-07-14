import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import *
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights=None,input_size=(512,512,1),dropout=0.75):
    inputs=Input(input_size)
    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    drop1=Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2=MaxPooling2D(pool_size=(2,2))(drop2)

    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    drop3=Dropout(dropout)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2))(drop3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5=Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5=Dropout(0.2)(conv5)

    up6=Conv2D(512,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6=concatenate([drop4,up6],axis=3)
    conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([drop3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([drop2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([drop1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3,1,activation='sigmoid')(conv9)

    model=Model(input=inputs,output=conv10)
    model.compile(optimizer=Adam(lr=5e-4),loss='categorical_crossentropy',metrics=['accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


