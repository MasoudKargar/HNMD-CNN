import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import os
from PIL import Image
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
K.set_image_data_format('channels_last')
import cv2
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#______________________X_train    &    y_train _______________________________________________________

def load_data_new(): 

    global  X_train ,y_train , X_test, y_test
    count_train_del=0
    count_test_del=0
    MainTrain_DIR1="Data/flair/"
    #MainTrain_DIR2="Data/seg/"    
    MainTrain_DIR3="Data/t1/"     
    MainTrain_DIR4="Data/t1ce/"
    MainTrain_DIR5="Data/t2/"

    X_train=[]
    y_train=[]
    
    for j in range(200):
        
        flair_array_5y=[]
        if j<100:
            k="HGG/"
            jk=j #1--100
            y=1
        else:    
            k="LGG/"
            jk=j-100 #1--100
            y=0

        mD1=MainTrain_DIR1+k+str(jk+1)+"/"
        #mD2=MainTrain_DIR2+k+str(jk+1)+"/"
        mD3=MainTrain_DIR3+k+str(jk+1)+"/"
        mD4=MainTrain_DIR4+k+str(jk+1)+"/"        
        mD5=MainTrain_DIR5+k+str(jk+1)+"/" 
        
        '''
        for m in range(30,100):
            mD1=mD1+str(m)+"/"
            mD2=mD2+str(m)+"/"
            mD3=mD3+str(m)+"/"
            mD4=mD4+str(m)+"/"       
            mD5=mD5+str(m)+"/"   
        '''
        
        for m in range(30,110):
                print ("******************************",m)
                array_5X=[]
                img2 = cv2.imread(mD1+os.listdir(mD1)[m])
                #print(mD1+os.listdir(mD1)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                #############################                 
                
                img2 = cv2.imread(mD3+os.listdir(mD3)[m])
                #print(mD3+os.listdir(mD3)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                ############################# 
                
                img2 = cv2.imread(mD4+os.listdir(mD4)[m])
                #print(mD4+os.listdir(mD4)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                #############################
                
                img2 = cv2.imread(mD5+os.listdir(mD5)[m])
                #print(mD5+os.listdir(mD5)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                y_train.append(y)
                #############################


                #print(numpy.shape(array_5X))
                #print(type(array_5X))  
                #print(len(array_5X))   
                array_5X= numpy.stack((array_5X),axis=0)
                array_5X=array_5X.reshape(248,248,3) 
                #print(type(array_5X))  
                #print(numpy.shape(array_5X)) 
                X_train.append(array_5X)  
    X_train= numpy.stack((X_train),axis=0)
    y_train= numpy.stack((y_train),axis=0) 
   
    X_train=X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5
    
    print("X_train  :  ",numpy.shape(X_train))
    print("X_train  :  ",type(X_train))        
    print("y_train",numpy.shape(y_train))
    print("y_train",type(y_train))  
        
    MainValid_DIR1="Data/flair/"
    #MainValid_DIR2="Data/seg/"    
    MainValid_DIR3="Data/t1/"     
    MainValid_DIR4="Data/t1ce/"
    MainValid_DIR5="Data/t2/"

    X_test=[]
    y_test=[]
   
    for j in range(100,180):
        
        flair_array_5y=[]
        if j<140:
            k="HGG/"
            jk=j #1--100
            y=1
        else:    
            k="LGG/"
            jk=j-140 #1--100
            y=0

        mD1=MainValid_DIR1+k+str(jk+1)+"/"
        #mD2=MainValid_DIR2+k+str(jk+1)+"/"
        mD3=MainValid_DIR3+k+str(jk+1)+"/"
        mD4=MainValid_DIR4+k+str(jk+1)+"/"        
        mD5=MainValid_DIR5+k+str(jk+1)+"/" 

        for m in range(30,110):

                array_5X=[]
                img2 = cv2.imread(mD1+os.listdir(mD1)[m])
                #print(mD1+os.listdir(mD1)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                #############################
                '''
                img2 = cv2.imread(mD2+os.listdir(mD2)[m])
                #print(mD2+os.listdir(mD2)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                '''
                #############################                    
                
                img2 = cv2.imread(mD3+os.listdir(mD3)[m])
                #print(mD3+os.listdir(mD3)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                ############################# 
                
                img2 = cv2.imread(mD4+os.listdir(mD4)[m])
                #print(mD4+os.listdir(mD4)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                #############################
                
                img2 = cv2.imread(mD5+os.listdir(mD5)[m])
                #print(mD5+os.listdir(mD5)[m])
                a,b,c=img2.shape
                img2=img2[int((a-192)/2):int((a+192)/2),int((b-192)/2):int((b+192)/2)]
                img2 = cv2.resize(img2, (124,124))
                a,b,c=img2.shape
                #print("____________________",a,b,c)           
                x2=numpy.array(img2)
                array_5X.append(x2)
                y_test.append(y)
                #############################


                #print(numpy.shape(array_5X))
                #print(type(array_5X))  
                #print(len(array_5X))   
                array_5X= numpy.stack((array_5X),axis=0)
                array_5X=array_5X.reshape(248,248,3) 
                #print(type(array_5X))  
                #print(numpy.shape(array_5X)) 
                X_test.append(array_5X)
    X_test= numpy.stack((X_test),axis=0)
    y_test= numpy.stack((y_test),axis=0) 
    
    X_test=X_test.astype('float32')
    X_test = (X_test - 127.5) / 127.5
    
    print("count_train_del:",count_train_del)    
    print("count_test_del:",count_test_del)   
    print("X_test  :  ",numpy.shape(X_test))
    print("X_test  :  ",type(X_test))        
    print("y_test",numpy.shape(y_test))
    print("y_test",type(y_test))        
            
load_data_new() 
 
input_shape = (248,248,3) 

epoch = 20
model = Sequential()
model.add(Dropout(0.2,input_shape=input_shape ))
model.add(Conv2D(64,(13,13) ,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2))
model.add(LeakyReLU(alpha=0.2))
model.add(BatchNormalization())

model.add(Conv2D(64,(9,9),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128,(7,7),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128,(5,5),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
#model.add(LSTM(200))
model.add(Dense(units=1,activation='sigmoid'))

adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy']) 
   
print(model.summary())

# checkpoint
filepath="weights-improvement_FINAL5-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor= 'accuracy' , verbose=1, save_best_only=True,
mode= max )
callbacks_list = [checkpoint]

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=2,callbacks=callbacks_list, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model_final5.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_final4.h5")
print("Saved model to disk")
print("______________________________________________________________________")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history[ 'accuracy' ])
plt.plot(history.history[ 'val_accuracy' ])
plt.title( 'model accuracy' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'upper left' )
plt.show()
# summarize history for loss
plt.plot(history.history[ 'loss' ])
plt.plot(history.history[ 'val_loss' ])
plt.title( 'model loss' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'upper left' )
plt.show()

'''
output
X_train  :   (16000, 248, 248, 3)
X_train  :   <class 'numpy.ndarray'>
y_train (16000,)
y_train <class 'numpy.ndarray'>

X_test  :   (6400, 248, 248, 3)
X_test  :   <class 'numpy.ndarray'>
y_test (6400,)
y_test <class 'numpy.ndarray'>
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dropout (Dropout)           (None, 248, 248, 3)       0

 conv2d (Conv2D)             (None, 236, 236, 64)      32512

 max_pooling2d (MaxPooling2D  (None, 117, 117, 64)     0
 )

 leaky_re_lu (LeakyReLU)     (None, 117, 117, 64)      0

 batch_normalization (BatchN  (None, 117, 117, 64)     256
 ormalization)

 conv2d_1 (Conv2D)           (None, 109, 109, 64)      331840

 max_pooling2d_1 (MaxPooling  (None, 53, 53, 64)       0
 2D)

 leaky_re_lu_1 (LeakyReLU)   (None, 53, 53, 64)        0

 dropout_1 (Dropout)         (None, 53, 53, 64)        0

 batch_normalization_1 (Batc  (None, 53, 53, 64)       256
 hNormalization)

 conv2d_2 (Conv2D)           (None, 47, 47, 128)       401536

 max_pooling2d_2 (MaxPooling  (None, 22, 22, 128)      0
 2D)

 leaky_re_lu_2 (LeakyReLU)   (None, 22, 22, 128)       0

 dropout_2 (Dropout)         (None, 22, 22, 128)       0

 batch_normalization_2 (Batc  (None, 22, 22, 128)      512
 hNormalization)

 conv2d_3 (Conv2D)           (None, 18, 18, 128)       409728

 max_pooling2d_3 (MaxPooling  (None, 8, 8, 128)        0
 2D)

 leaky_re_lu_3 (LeakyReLU)   (None, 8, 8, 128)         0

 dropout_3 (Dropout)         (None, 8, 8, 128)         0

 batch_normalization_3 (Batc  (None, 8, 8, 128)        512
 hNormalization)

 conv2d_4 (Conv2D)           (None, 6, 6, 256)         295168

 max_pooling2d_4 (MaxPooling  (None, 2, 2, 256)        0
 2D)

 leaky_re_lu_4 (LeakyReLU)   (None, 2, 2, 256)         0

 dropout_4 (Dropout)         (None, 2, 2, 256)         0

 batch_normalization_4 (Batc  (None, 2, 2, 256)        1024
 hNormalization)

 flatten (Flatten)           (None, 1024)              0

 dense (Dense)               (None, 256)               262400

 dense_1 (Dense)             (None, 128)               32896

 dense_2 (Dense)             (None, 64)                8256

 dense_3 (Dense)             (None, 1)                 65

=================================================================
Total params: 1,776,961
Trainable params: 1,775,681
Non-trainable params: 1,280
_________________________________________________________________
None
WARNING:tensorflow:ModelCheckpoint mode <built-in function max> is unknown, fallback to auto mode.
Epoch 1/20

Epoch 1: accuracy improved from -inf to 0.86881, saving model to weights-improvement_FINAL5-01-0.87.hdf5
8000/8000 - 10805s - loss: 0.3222 - accuracy: 0.8688 - val_loss: 0.2269 - val_accuracy: 0.9008 - 10805s/epoch - 1s/step
Epoch 2/20

Epoch 2: accuracy improved from 0.86881 to 0.92750, saving model to weights-improvement_FINAL5-02-0.93.hdf5
8000/8000 - 10481s - loss: 0.1953 - accuracy: 0.9275 - val_loss: 0.0812 - val_accuracy: 0.9664 - 10481s/epoch - 1s/step
Epoch 3/20

Epoch 3: accuracy improved from 0.92750 to 0.95275, saving model to weights-improvement_FINAL5-03-0.95.hdf5
8000/8000 - 11849s - loss: 0.1324 - accuracy: 0.9528 - val_loss: 0.0983 - val_accuracy: 0.9633 - 11849s/epoch - 1s/step
Epoch 4/20

Epoch 4: accuracy improved from 0.95275 to 0.96288, saving model to weights-improvement_FINAL5-04-0.96.hdf5
8000/8000 - 10538s - loss: 0.1078 - accuracy: 0.9629 - val_loss: 0.0208 - val_accuracy: 0.9967 - 10538s/epoch - 1s/step
Epoch 5/20

Epoch 5: accuracy improved from 0.96288 to 0.97600, saving model to weights-improvement_FINAL5-05-0.98.hdf5
8000/8000 - 10581s - loss: 0.0742 - accuracy: 0.9760 - val_loss: 0.0254 - val_accuracy: 0.9927 - 10581s/epoch - 1s/step
Epoch 6/20

Epoch 6: accuracy improved from 0.97600 to 0.97719, saving model to weights-improvement_FINAL5-06-0.98.hdf5
8000/8000 - 10544s - loss: 0.0676 - accuracy: 0.9772 - val_loss: 0.0546 - val_accuracy: 0.9906 - 10544s/epoch - 1s/step 
Epoch 7/20

Epoch 7: accuracy improved from 0.97719 to 0.97938, saving model to weights-improvement_FINAL5-07-0.98.hdf5
8000/8000 - 10777s - loss: 0.0598 - accuracy: 0.9794 - val_loss: 0.0103 - val_accuracy: 0.9978 - 10777s/epoch - 1s/step
Epoch 8/20

Epoch 8: accuracy improved from 0.97938 to 0.98106, saving model to weights-improvement_FINAL5-08-0.98.hdf5
8000/8000 - 10647s - loss: 0.0549 - accuracy: 0.9811 - val_loss: 0.0247 - val_accuracy: 0.9973 - 10647s/epoch - 1s/step
Epoch 9/20

Epoch 9: accuracy improved from 0.98106 to 0.98431, saving model to weights-improvement_FINAL5-09-0.98.hdf5
8000/8000 - 10604s - loss: 0.0482 - accuracy: 0.9843 - val_loss: 0.0129 - val_accuracy: 0.9973 - 10604s/epoch - 1s/step
Epoch 10/20

Epoch 10: accuracy improved from 0.98431 to 0.98750, saving model to weights-improvement_FINAL5-10-0.99.hdf5
8000/8000 - 10666s - loss: 0.0433 - accuracy: 0.9875 - val_loss: 0.0282 - val_accuracy: 0.9930 - 10666s/epoch - 1s/step
Epoch 11/20

Epoch 11: accuracy improved from 0.98750 to 0.98819, saving model to weights-improvement_FINAL5-11-0.99.hdf5
8000/8000 - 10576s - loss: 0.0402 - accuracy: 0.9882 - val_loss: 0.0321 - val_accuracy: 0.9970 - 10576s/epoch - 1s/step
Epoch 12/20

Epoch 12: accuracy improved from 0.98819 to 0.98869, saving model to weights-improvement_FINAL5-12-0.99.hdf5
8000/8000 - 10528s - loss: 0.0367 - accuracy: 0.9887 - val_loss: 0.0378 - val_accuracy: 0.9975 - 10528s/epoch - 1s/step
Epoch 13/20

Epoch 13: accuracy improved from 0.98869 to 0.98919, saving model to weights-improvement_FINAL5-13-0.99.hdf5
8000/8000 - 10480s - loss: 0.0371 - accuracy: 0.9892 - val_loss: 0.1447 - val_accuracy: 0.9944 - 10480s/epoch - 1s/step
Epoch 14/20

Epoch 14: accuracy improved from 0.98919 to 0.99119, saving model to weights-improvement_FINAL5-14-0.99.hdf5
8000/8000 - 10502s - loss: 0.0285 - accuracy: 0.9912 - val_loss: 0.0919 - val_accuracy: 0.9798 - 10502s/epoch - 1s/step
Epoch 15/20

Epoch 15: accuracy improved from 0.99119 to 0.99262, saving model to weights-improvement_FINAL5-15-0.99.hdf5
8000/8000 - 10494s - loss: 0.0307 - accuracy: 0.9926 - val_loss: 0.0139 - val_accuracy: 0.9958 - 10494s/epoch - 1s/step
Epoch 16/20

Epoch 16: accuracy did not improve from 0.99262
8000/8000 - 10393s - loss: 0.0293 - accuracy: 0.9916 - val_loss: 0.0011 - val_accuracy: 0.9997 - 10393s/epoch - 1s/step
Epoch 17/20

Epoch 17: accuracy improved from 0.99262 to 0.99275, saving model to weights-improvement_FINAL5-17-0.99.hdf5
8000/8000 - 10427s - loss: 0.0301 - accuracy: 0.9927 - val_loss: 0.0175 - val_accuracy: 0.9987 - 10427s/epoch - 1s/step
Epoch 18/20

Epoch 18: accuracy did not improve from 0.99275
8000/8000 - 11558s - loss: 0.0311 - accuracy: 0.9912 - val_loss: 0.0079 - val_accuracy: 0.9966 - 11558s/epoch - 1s/step
Epoch 19/20

Epoch 19: accuracy improved from 0.99275 to 0.99406, saving model to weights-improvement_FINAL5-19-0.99.hdf5
8000/8000 - 10981s - loss: 0.0244 - accuracy: 0.9941 - val_loss: 0.0338 - val_accuracy: 0.9891 - 10981s/epoch - 1s/step
Epoch 20/20

Epoch 20: accuracy did not improve from 0.99406
8000/8000 - 10639s - loss: 0.0226 - accuracy: 0.9935 - val_loss: 0.0030 - val_accuracy: 0.9994 - 10639s/epoch - 1s/step
200/200 [==============================] - 966s 5s/step - loss: 0.0030 - accuracy: 0.9994
Accuracy: 99.94%
Saved model to disk
'''