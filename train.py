import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import get_file
import sys
from datetime import datetime

epochs = 20

batch = 128

model_type = 'Condition'

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

# print(val_df.head())
# sys.exit()

# lblmapsub = {'Abotinado': 0, 'Adventure': 1, 'Anabela': 2, 'Ankle Boot': 3, 'Birken': 4, 'Blaqueada': 5,
#              'Cano Baixo': 6, 'Cano Curto': 7, 'Cano Longo': 8, 'Cano Medio': 9, 'Casual': 10, 'Chanel': 11, 
#              'Chelsea': 12, 'Chunky': 13, 'Colegial': 14, 'Com Atacador': 15,'Coturno': 16, 'De Dedo': 17, 
#              'Esportivo': 18, 'Fechado': 19, 'Flatform': 20, 'Huarache': 21, 'Jogging': 22, 'Knit': 23, 'Loafer': 24, 
#              'Mary Jane': 25, 'Mocassim': 26, 'Montaria': 27, 'Mule': 28, 'Over The Knee': 29, 'Oxford': 30, 'Papete': 31,
#              'Peep Toe': 32, 'Plataforma': 33, 'Rasteira': 34, 'Salto': 35, 'Sapatilha': 36, 'Scarpin': 37,  'Slip On': 38,
#              'Sneaker': 39, 'Sport Sandal': 40, 'Tamanco': 41, 'Tamanco De Dedo': 42, 'Upper': 43}

lblmapsub = {'Cano Curto': 0, 'Cano Medio': 1, 'Cano Longo': 2, 'Slip On': 3, 'De Dedo': 4, 'Sapatilha': 5,
             'Mule': 6, 'Rasteira': 7, 'Scarpin': 8, 'Esportivo': 9, 'Casual': 10, 'Cano Baixo': 11, 'Flatform': 12,
             'Salto': 13,  'Chunky': 14,  'Jogging': 15,  'Coturno': 16,  'Anabela': 17}
lblmapmaster = {'Bota': 0, 'Sandalia': 1, 'Sapato': 2, 'Tenis': 3}

#Map classes
train_df['category'].replace(lblmapmaster,inplace=True)
train_df['subcategory'].replace(lblmapsub,inplace=True)

val_df['category'].replace(lblmapmaster,inplace=True)
val_df['subcategory'].replace(lblmapsub,inplace=True)

#Convert the 3 labels to one hots in train, test, val
onehot_master = to_categorical(train_df['category'].values)
train_df['CategoryOneHot'] = onehot_master.tolist()

onehot_master = to_categorical(train_df['subcategory'].values)
train_df['SubCategoryOneHot'] = onehot_master.tolist()

onehot_sub = to_categorical(val_df['category'].values)
val_df['CategoryOneHot'] = onehot_sub.tolist()

onehot_sub = to_categorical(val_df['subcategory'].values)
val_df['SubCategoryOneHot'] = onehot_sub.tolist()

# print(train_df.head(5))

# print(val_df.head(5))
# sys.exit()

direc = './img/'
target_size=(150, 200)
TODAY = datetime.now().strftime("%Y-%m-%d-%H%M%S") # str(datetime.date(datetime.now()))

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   rotation_range=7,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)
val_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

def get_flow_from_dataframe(g, dataframe,image_shape=target_size,batch_size=batch):
    while True:
        x_1 = g.next()

        yield [x_1[0], x_1[1][0], x_1[1][1]], x_1[1]

def train_recurrent(label, model,cbks):
    # model.load_weights(weights_path, by_name=True)
    train = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=direc,
        x_col="image",
        y_col=['CategoryOneHot','SubCategoryOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')
    val = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=direc,
        x_col="image",
        y_col=['CategoryOneHot','SubCategoryOneHot'],
        target_size=target_size,
        batch_size=batch,
        class_mode='multi_output')

    train_generator = get_flow_from_dataframe(train,dataframe=train_df,image_shape=target_size,batch_size=batch)
    val_generator = get_flow_from_dataframe(val,dataframe=val_df,image_shape=target_size,batch_size=batch)
    try:
        print("Start training")
        STEP_SIZE_TRAIN = train.n // train.batch_size
        STEP_SIZE_VALID = val.n // val.batch_size
        history = model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=val_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=cbks)
        print("Finished training")
        #Save training as csv
        pd.DataFrame.from_dict(history.history).to_csv("./history/"+label+"_"+str(epochs)+"_epochs_"+TODAY+'_normal.csv',index=False)

        # summarize history for Accuracy
        plt.plot(history.history['master_output_categorical_accuracy'])
        plt.plot(history.history['val_master_output_categorical_accuracy'])
        plt.plot(history.history['sub_output_categorical_accuracy'])
        plt.plot(history.history['val_sub_output_categorical_accuracy'])
        plt.title('Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train master', 'Val master', 'Train sub', 'Val sub'], loc='upper left')
        plt.show()
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_accuracy_normal.png", bbox_inches='tight')
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['master_output_loss'])
        plt.plot(history.history['val_master_output_loss'])
        plt.plot(history.history['sub_output_loss'])
        plt.plot(history.history['val_sub_output_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train master', 'Val master', 'Train sub', 'Val sub'], loc='upper left')
        plt.show()
        plt.savefig("./plots/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_loss_normal.png", bbox_inches='tight')
        plt.clf()


    except ValueError as v:
        print(v)

    # Saving the weights in the current directory
    model.save_weights("./weights/"+label+"_"+str(epochs)+"_epochs_"+TODAY+"_normal.h5")                                        

from cnn import Train
train = Train(model_type)
model = train.model
cbks = train.cbks
train_recurrent(model_type,model,cbks)