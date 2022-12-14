import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout, Add, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.constraints import Constraint
import numpy as np
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import Lambda
import sys
import json 

print(tf.__version__ )

def saveParameters(content):
    file = open("parameters.json", "w")
    file.write(content + "\r\n")
    file.close()

# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class NonNegUnitNorm(Constraint):
    '''Enforces all weight elements to be non-0 and each column/row to be unit norm'''
    def __init__(self, axis=1):
        self.axis=axis
    def __call__(self, w):
        w = w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())
        return w / (
            K.epsilon() + K.sqrt(
                math_ops.reduce_sum(
                    math_ops.square(w), axis=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis}

class Train:
    '''This model is based off of VGG16 with the addition of BatchNorm layers and then branching '''

    def __init__(self,label):
        self.master_classes=4
        self.sub_classes=18 #44
        
        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        input_image = Input(shape=(150,200,3),name="InputImg")

        #4 element vector for the masterCategory types
        input_master = Input(shape=(self.master_classes))

        #21 element vector for the subCategory types
        input_sub = Input(shape=(self.sub_classes))

        #Layers are named to be same as VGG16

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- block 5 masterCategory ---
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_mas')(x)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_mas')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_mas')(x1)
        x1 = BatchNormalization()(x1)

        #--- masterCategory prediction branch ---
        c_1_bch = Flatten(name='c1_flatten')(x1)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.master_classes, activation='softmax', name='master_output')(c_1_bch)

        #--- block 5 subCategory ---
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_sub')(x)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_sub')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_sub')(x2)
        x2 = BatchNormalization()(x2)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x2)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)

        #--- masterCategory conditioning for subCategory branch ---
        c_1_condition = Dense(self.sub_classes, activation=None, use_bias=False, kernel_constraint=NonNegUnitNorm(),kernel_initializer=Zeros(),name='c_1_condition')(input_master)
        c_2_raw = Dense(self.sub_classes, activation='relu', name='c_2_raw')(c_2_bch)
        preds_features = Add()([c_1_condition,c_2_raw])

        #c_2_pred = Dense(self.sub_classes, activation='softmax', name='sub_output')(preds_features)
        c_2_pred = Softmax(name='sub_output')(preds_features)

        # Model to be trained
        #modelC = Model(inputs=[input_image,input_master,input_sub], outputs=z)

            # trainable_params = tf.keras.backend.count_params(model.trainable_weights)
            # print("Trainable paramaters: "+str(trainable_params))

        # print(model.summary())
        # Compiling the model
        # KEras will automatically use categorical accuracy when accuracy is used.

        model = Model(
            inputs=[input_image,input_master,input_sub],
            outputs=[c_1_pred, c_2_pred],
            name="Condition_CNN")

        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        #trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("===== Trainable paramaters: "+str(trainable_params))

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy",
        }
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss=losses, metrics=['categorical_accuracy', f1_m, precision_m, recall_m])
        # model.compile(optimizer=Adam(lr=0.001), loss=losses, metrics=['categorical_accuracy', f1_m, precision_m, recall_m])
        # model.compile(optimizer=RMSprop(lr=0.001, momentum=0.9), loss=losses, metrics=['categorical_accuracy', f1_m, precision_m, recall_m])

        saveParameters(json.dumps(model.get_config(), indent = 4))

        checkpoint = ModelCheckpoint("./weights/"+label+"_best_weights_tf_50_nesterov.h5", monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        self.cbks = [checkpoint]
        self.model = model


class Test:
    '''One parameter model which is a keras model'''

    def __init__(self,label):

        self.master_classes=4
        self.sub_classes=18

        '''Three inputs to model for training, image, labels ,labels for teacher forcing'''
        input_image = Input(shape=(150,200,3),name="InputImg")

        #Layers are named to be same as VGG16

        #will need to center and scale data.

        #--- block 1 ---
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        #--- block 2 ---
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        #--- block 3 ---
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        #--- block 4 ---
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        #--- block 5 masterCategory ---
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_mas')(x)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_mas')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_mas')(x1)
        x1 = BatchNormalization()(x1)

        #--- masterCategory prediction branch ---
        c_1_bch = Flatten(name='c1_flatten')(x1)
        c_1_bch = Dense(256, activation='relu', name='c1_fc_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_bch = Dense(256, activation='relu', name='c1_fc2_mas')(c_1_bch)
        c_1_bch = BatchNormalization()(c_1_bch)
        c_1_bch = Dropout(0.5)(c_1_bch)
        c_1_pred = Dense(self.master_classes, activation='softmax', name='master_output')(c_1_bch)

        #--- block 5 subCategory ---
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1_sub')(x)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2_sub')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3_sub')(x2)
        x2 = BatchNormalization()(x2)

        #--- coarse 2 branch ---
        c_2_bch = Flatten(name='c2_flatten')(x2)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)
        c_2_bch = Dense(1024, activation='relu', name='c2_fc2_sub')(c_2_bch)
        c_2_bch = BatchNormalization()(c_2_bch)
        c_2_bch = Dropout(0.5)(c_2_bch)

        #--- masterCategory conditioning for subCategory branch ---
        c_1_condition = Dense(self.sub_classes, activation=None, use_bias=False, kernel_constraint=NonNegUnitNorm(),name='c_1_condition')(c_1_pred)
        c_2_raw = Dense(self.sub_classes, activation='relu', name='c_2_raw')(c_2_bch)
        preds_features = Add()([c_1_condition,c_2_raw])
        c_2_pred = Softmax(name='sub_output')(preds_features)

        model = Model(
            inputs=[input_image],
            outputs=[c_1_pred, c_2_pred],
            name="Condition_CNN")

        losses = {
            "master_output": "categorical_crossentropy",
            "sub_output": "categorical_crossentropy"
        }

        trainable_params= np.sum([K.count_params(w) for w in model.trainable_weights])
        #trainable_params = tf.keras.backend.count_params(model.trainable_weights)
        print("Trainable paramaters: "+str(trainable_params))

        model.compile(optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), loss=losses,
                      metrics=['categorical_accuracy', f1_m, precision_m, recall_m])
                      
        checkpoint = ModelCheckpoint("./weights/"+label+"_best_weights_tf_50_rms.h5", monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True,mode='auto')
        self.cbks = [checkpoint]
        self.model = model