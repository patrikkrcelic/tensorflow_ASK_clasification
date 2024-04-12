import numpy as np
import matplotlib.pyplot as plt

# defining the function which reads in the images and performs the preprocessing

def imrd(file, ndim):
    f=open(file,'r')

    im=np.zeros((ndim,ndim))

    br1=0

    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        br2=0
        for s in inner_list:
            if s != '':
               im[br1,br2]=float(s)
               br2=br2+1
        br1=br1+1

    im2=np.sort(np.reshape(im,ndim*ndim))
    immax=im2[int(np.round(ndim*ndim*0.95))]

    br1=0
    for i in im:
        I=np.where(i > immax)
        im[br1,I]=immax
        I=np.where(i < 0.0)
        im[br1,I]=1
        br1=br1+1

    im=im/immax
    f.close()
    return im



# preparing the data for the analysis

import tensorflow as tf
from tensorflow.keras import datasets, layers, models



# data directory
datdir='/media/patrik/HD/ASK_files/clasification_files/multiple_clases_first/'
# directory to save the model to
datdir_model=''

ndim=256
ntrain=800
nval=200


X=np.zeros((ntrain,ndim,ndim))
X_val=np.zeros((nval,ndim,ndim))

ncat=5


f=open('imfil_multi_list_2.txt','r')

brtrain=0
brval=0
#in_test=[0]
ytrain=np.zeros((ntrain,5))
yval=np.zeros((nval,5))

for line in f:
    inner_list = [elt.strip() for elt in line.split(' ')]
    brblock=0
    namenum=np.cumsum(np.ones(10))-1
    if brval < nval:
        yval[brval,0]=float(inner_list[1])
        yval[brval,1]=float(inner_list[2])
        yval[brval,2]=float(inner_list[3])
        yval[brval,3]=float(inner_list[4])
        yval[brval,4]=float(inner_list[5])
        namenum=float(inner_list[0])+namenum

        for i in namenum:
            if i < 10:
               nmnum='0000'+str(i)
            elif i>=10 and i<100:
               nmnum='000'+str(i)
            elif i>=100 and i<1000:
               nmnum='00'+str(i)
            elif i>=1000 and i<10000:
               nmnum='0'+str(i)
            else:
               nmnum=str(i)
            nmnum=nmnum[:-2]
            X_val[brval,:,:,brblock]=imrd(datdir+'ask1_'+nmnum+'.txt', ndim)
            brblock=brblock+1
        brval=brval+1
    else:
        ytrain[brtrain,0]=float(inner_list[1])
        ytrain[brtrain,1]=float(inner_list[2])
        ytrain[brtrain,2]=float(inner_list[3])
        ytrain[brtrain,3]=float(inner_list[4])
        ytrain[brtrain,4]=float(inner_list[5])
        namenum=float(inner_list[0])+namenum

        for i in namenum:
            if i < 10:
               nmnum='0000'+str(i)
            elif i>=10 and i<100:
               nmnum='000'+str(i)
            elif i>=100 and i<1000:
               nmnum='00'+str(i)
            elif i>=1000 and i<10000:
               nmnum='0'+str(i)
            else:
               nmnum=str(i)
            nmnum=nmnum[:-2]
            X[brtrain,:,:,brblock]=imrd(datdir+'ask1_'+nmnum+'.txt', ndim)
            brblock=brblock+1
        brtrain=brtrain+1

f.close()


y = tf.constant(ytrain, dtype=tf.float32)
y_val = tf.constant(yval, dtype=tf.float32)


from tensorflow.keras.utils import to_categorical
y = to_categorical(in_train[1:len(in_train)]-1)
y_val = to_categorical(in_val[1:len(in_train)]-1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# build the model

fs = 3 # filter_size

# Define layers (named, so we can nab them later)
inputs = layers.Input(shape=(ndim, ndim, 10))
poolin = layers.MaxPooling2D((2, 2))(inputs)
conv1A = layers.Conv2D(16, (fs, fs),activation='relu', padding="same",strides=(1,1))(poolin)
conv1A = layers.Conv2D(16, (fs, fs), activation='relu', padding="same",strides=(1,1))(poolin)
dropout1 = layers.Dropout(0.4)(conv1A)
conv1B = layers.Conv2D(16, (fs, fs), activation='relu', padding="same",strides=(1,1))(conv1A)
pool1 = layers.MaxPooling2D((2, 2))(conv1A)
bnorm1 = layers.BatchNormalization()(pool1)
conv2A = layers.Conv2D(32, (fs, fs) ,activation='relu', padding="same")(pool1)
dropout2 = layers.Dropout(0.4)(conv2A)
conv2B = layers.Conv2D(32, (fs, fs), activation='relu', padding="same")(conv2A)
pool2 = layers.MaxPooling2D((2, 2))(conv2A)
bnorm2 = layers.BatchNormalization()(pool2)
conv3A = layers.Conv2D(64, (fs, fs), activation='relu', padding="same")(pool2)
dropout3 = layers.Dropout(0.4)(conv3A)
conv3B = layers.Conv2D(64, (fs, fs), activation='relu', padding="same")(conv3A)
pool3 = layers.MaxPooling2D((2,2))(conv3A)
bnorm3 = layers.BatchNormalization()(pool3)
flatten1 = layers.Flatten()(pool3)
dense1 = layers.Dense(64,activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(flatten1)
dropout4 = layers.Dropout(0.4)(dense1)
dense2 = layers.Dense(32,activation='relu')(dense1)
dropout5 = layers.Dropout(0.4)(dense2)
dense3 = layers.Dense(5,activation='sigmoid')(dense2)

model = models.Model(inputs, dense3)

model.summary()

opt = tf.keras.optimizers.SGD(learning_rate=0.0001)

model.compile(optimizer=opt,
              metrics=['accuracy'],
              loss='binary_crossentropy')



# model training
with tf.device('/gpu:0'):
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

convergence(history)

predictions = model.predict(X_test)

# testing

br=0
dobri=0
losi_nema=0
losi_ima=0

for pred in predictions:
    if pred > 0.8:
        if y_test[br] == 1:
            dobri += 1
        else:
            losi_nema += 1
    else:
        if y_test[br] == 1:
            losi_ima += 1

    br += 1

print(br)
print(dobri)
print(losi_nema)
print(losi_ima)

# function for ploting convergence during training process
def convergence(history):

    history = history.history

    loss = history["loss"]
    val_loss = history["val_loss"]
    nepochs = len(loss)

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    plt.plot(np.arange(nepochs), loss, "k-", label="training loss")
    plt.plot(np.arange(nepochs), val_loss, "k--", label="validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(frameon=False)
    plt.show()

    plt.plot(np.arange(nepochs), accuracy, "k-", label="training accuracy")
    plt.plot(np.arange(nepochs), val_accuracy, "k--", label="validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(frameon=False)
    plt.show()

convergence(history)

#save the model
model.save(datdir_model)
