import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load cifar10 dataset
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

# Normalize pixel values to [0,1]
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# One-Hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

# Create datagenerator
datagen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


# Fit the generator to the training data
datagen.fit(X_train)


def create_model():
    model = models.Sequential()
    
    
    # Convolutional Layer 1
    model.add(layers.Input(shape=(32,32,3)))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    
    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,activation='softmax'))
    
    
    
    return model



model = create_model()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Train the model using augmented data generator
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(X_test,y_test),
    steps_per_epoch= X_train.shape[0] // 64
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")


plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()



plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()