import random
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

#from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

import h5py



#Load Dataset
hf = h5py.File('/home/aditya.vijaykumar/work/nikita/data/trainingsetv1d0.h5', 'r')

#Creat List to Store All Glitches
glitch_set=[]
for glitch in hf.keys():
	glitch_set.append(glitch)


img_size='0.5.png'
input_shape=(140, 170, 1)
batch_size =30


#Function to:
#	Count Number of Images in each Set, e.g, Train, Validation, Test
#	Creates Image Set
def preprocessX (data, glitch_set, img_size):

	image_count=0

	for glitch in glitch_set:
		length = len(hf[glitch][data].keys())
		image_count=image_count+length

	X=np.zeros((image_count, 140, 170))
	Y=[]
	i = 0 

	for glitch in glitch_set:
		for image in hf[glitch][data].keys():
			img = hf[glitch][data][image][img_size][0][:] 
			X[:][:][i] = img
			Y.append(glitch)
			i = i+1 

  	 #Reshape for Input to CNN
	X=X.reshape((image_count, 140, 170, 1))

	return X, Y



#Integer Encodes Labels
def preprocessY (lst):
	integer_encoded = label_encoder.fit_transform(np.array(lst))
	trainY=to_categorical(integer_encoded)
	return trainY



#Define Model
def create_model():

	model=Sequential()
	model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(256, activation='relu'))
	model.add(Dense(22, activation='softmax'))
	print(model.summary())
	model.compile(loss=CategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=['accuracy'])
	return model



 # Create empty arrays to contain batch of features and labels#
def gen(features, labels, batch_size):
 batch_features = np.zeros((batch_size, 140, 170))
 batch_labels = np.zeros((batch_size,22))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choices(range(len(labels)),k=batch_size)
     batch_features = features[index]
     batch_labels = labels[index]
     batch_features = batch_features.reshape(batch_size, 140, 170, 1)
   yield batch_features, batch_labels



trainX, trainY=preprocessX('train', glitch_set, img_size)
testX, testY=preprocessX('test', glitch_set, img_size)
validationX, validationY=preprocessX('validation', glitch_set, img_size)
trainY=preprocessY(trainY)
testY=preprocessY(testY)
validationY=preprocessY(validationY)


model=create_model()

checkpoint_path = '/home/aditya.vijaykumar/work/nikita/checkpoints/entire/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, save_weights_only=True, period=1)

model.save_weights(checkpoint_path.format(epoch=0))

#model.load_weights('/home/aditya.vijaykumar/work/nikita/checkpoints/entire/cp-0004.ckpt')

model.fit_generator(gen(trainX, trainY, batch_size),	
                    steps_per_epoch=trainY.shape[0]//batch_size, 
                    epochs=10, 
                    validation_data=gen(validationX, validationY, batch_size),
                    validation_steps=validationY.shape[0]//batch_size,
                    callbacks=[cp_callback] )

model.save("/home/aditya.vijaykumar/work/nikita/checkpoints/entire/model.h5")
