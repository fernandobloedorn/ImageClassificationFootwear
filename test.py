import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime
from os import path
import tensorflow.keras.backend as K

#positional arguments required for test
model_type = 'Condition'
batch = 128
weights_path = './weights/Condition_best_weights.h5'

test_df = pd.read_csv("test.csv")

#check if results csv exists. if it does not then create it. append each result test run as a new label 
#row contains the label (BCNN, Recurrent etc), the weights path, the three test accuracies (or 1 if it is a baseline cnn model)
# and the timestamp when it was tested
exists = path.exists("./testing/test_results.csv")
if(not exists):
    dtypes = np.dtype([
          ('Model', str),
          ('Weights Path', str),
          ('Category Accuracy %', np.float64),
          ('SubCategory Accuracy %', np.float64),
          ('Trainable params', np.float64),
          ('Timestamp', np.datetime64),
          ])
    #df = pd.DataFrame(columns=['Model','Weights Path', 'masterCategory Accuracy %','subCategory Accuracy %','articleType Accuracy %','Timestamp'])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data)
else:
    types = {'Model':str,'Weights Path': str, 'Category Accuracy %': np.float64, 'SubCategory Accuracy %': np.float64, 'Trainable params': np.float64}
    df = pd.read_csv("./testing/test_results.csv",dtype=types, parse_dates=['Timestamp'])

lblmapsub = {'Cano Curto': 0, 'Cano Medio': 1, 'Cano Longo': 2, 'Slip On': 3, 'De Dedo': 4, 'Sapatilha': 5,
             'Mule': 6, 'Rasteira': 7, 'Scarpin': 8, 'Esportivo': 9, 'Casual': 10, 'Cano Baixo': 11, 'Flatform': 12,
             'Salto': 13,  'Chunky': 14,  'Jogging': 15,  'Coturno': 16,  'Anabela': 17}
lblmapmaster = {'Bota': 0, 'Sandalia': 1, 'Sapato': 2, 'Tenis': 3}

#Map classes
test_df['category'].replace(lblmapmaster,inplace=True)
test_df['subcategory'].replace(lblmapsub,inplace=True)

#Convert the 3 labels to one hots in train, test, val
onehot_master = to_categorical(test_df['category'].values)
test_df['categoryOneHot'] = onehot_master.tolist()

onehot_master = to_categorical(test_df['subcategory'].values)
test_df['subCategoryOneHot'] = onehot_master.tolist()

#----------globals---------
direc = './img/'
target_size=(150,200)
TODAY = str(datetime.date(datetime.now()))

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

def test_multi(label, model):
    model.load_weights(weights_path, by_name=True)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=direc,
        x_col="image",
        y_col=['categoryOneHot','subCategoryOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    #x can be a generator returning retunring (inputs, targets)
    #if x is a generator y should not be specified
    score = model.evaluate(x=test_generator, steps=STEP_SIZE_TEST)

    print(f"score is {score}")
    return score

score = 0
params = 0
masterCategory_accuracy = np.nan
subCategory_accuracy = np.nan

from cnn import Test
model = Test(model_type).model
score = test_multi(model_type, model)
params= np.sum([K.count_params(w) for w in model.trainable_weights])
masterCategory_accuracy = score[3]
subCategory_accuracy = score[4]

df.loc[df.index.max()+1] = [model_type, weights_path, masterCategory_accuracy, subCategory_accuracy, params,np.datetime64('now')]
df.to_csv("./testing/test_results.csv", index=False)
