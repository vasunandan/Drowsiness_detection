import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten               #libraries
from keras.models import Sequential

from keras.models import load_model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(25,25,1)))
model.add(MaxPool2D(pool_size=(1,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(1,1)))                                 # structure of the model
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(70,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

batch_size = 2
train_gen_proc = ImageDataGenerator(
                rotation_range=20,
                shear_range=0.2,
                zoom_range=0.3,
                horizontal_flip=True,
                rescale=1./255,
                validation_split=0.1
                )


'''for i in range(1,76):                                                      # Data augmentataion of a single image file
    img = load_img(f'data/save_img_{i}.jpg')
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    j = 0
    for batch in train_gen_proc.flow(x,batch_size=1,save_to_dir='generated_photos',save_prefix=f'gen({i})',save_format='jpeg'):
        j +=1
        if j>25:
            break'''
train_gen = train_gen_proc.flow_from_directory('data',target_size=(25,25),class_mode='binary',color_mode='grayscale',batch_size=batch_size,shuffle=True,subset='training')
valid_gen = train_gen_proc.flow_from_directory('data',target_size=(25,25),class_mode='binary',color_mode='grayscale',batch_size=batch_size,subset='validation')


model.fit(train_gen,steps_per_epoch=len(train_gen)//batch_size,epochs=10,verbose=1)                   # training the model

model.save('models_drows/model_eye_detector.h5', overwrite=True) # save the model
