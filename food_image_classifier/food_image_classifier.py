# import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# define dataset paths
TRAIN_PATH = 'train'
VALID_PATH = 'valid'
TEST_PATH = 'test'

# define image dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224

# define number of classes
NUM_CLASSES = 4

# define batch size
BATCH_SIZE = 32

# define number of epochs
NUM_EPOCHS = 10

# define checkpoints directory
CHECKPOINT_PATH = 'checkpoints'

# create checkpoints directory if it doesn't exist
if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# define image data generators with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# load training, validation, and test datasets
train_dataset = train_datagen.flow_from_directory(directory=TRAIN_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')

valid_dataset = valid_datagen.flow_from_directory(directory=VALID_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')

test_dataset = test_datagen.flow_from_directory(directory=TEST_PATH,
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

# define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# define callbacks for model checkpoint and early stopping
checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH+'/best_model.h5',
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max')

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=3)

# train the model
history = model.fit(train_dataset,
                    epochs=NUM_EPOCHS,
                    validation_data=valid_dataset,
                    callbacks=[checkpoint_callback, early_stop_callback])

# save the trained model
model.save('food_model.h5')