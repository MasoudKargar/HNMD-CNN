import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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
        

    #input("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@               ??????")
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

    #input("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  ??????????")    
    
load_data_new() 

 
input_shape = (248,248,3) 

epoch = 20
model = Sequential()
model.add(Dropout(0.2,input_shape=input_shape ))
model.add(Conv2D(64,(13,13) ,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(4,4),strides=2)) #if stride not given it equal to pool filter size
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

model.load_weights("model_final5.h5")
adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy']) 

print("Created model and loaded weights from file")    
print(model.summary())



# Fit the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=2,callbacks=callbacks_list, verbose=2)
###########
# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=1)
_, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('Train: %.4f, Test: %.4f' % (train_acc, test_acc))



# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=1)
# predict crisp classes for test set

#yhat_classes = model.predict_classes(X_test, verbose=0)
yhat_classes = (model.predict(X_test) > 0.5).astype("int32")
print(numpy.shape(yhat_classes))
print('classes:',yhat_classes[:10])
print('classes:',y_test[:10])

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(y_test, yhat_classes)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)

'''
Train: 0.9985, Test: 0.9994
200/200 [==============================] - 956s 5s/step
(6400, 1)
classes: [[1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]]
classes: [1 1 1 1 1 1 1 1 1 1]
Accuracy: 0.999375
Precision: 0.998752
Recall: 1.000000
F1 score: 0.999375
Cohens kappa: 0.998750
ROC AUC: 0.999999
[[3196    4]
 [   0 3200]]
 '''




