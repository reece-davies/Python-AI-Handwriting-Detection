# Description: Classifies the MNIST handwritten digit images as number 0-9
# https://www.youtube.com/watch?v=kOFUQB7u5Ck&list=WL&index=6&t=301s

# Then use the guide for making a drawing application so it can predict what you have drawn
# https://www.youtube.com/watch?v=WVuMCawju-g&list=PLujzF9oKZSaTU_nEN9cRjMeEdhcDQT7sh&index=77&t=86s

# Additionally, we could make our own images using gimp and then run the ANN to predict what number they are:
# https://www.youtube.com/watch?v=E6CxY8gspfQ


# Download pip (or Anaconda), and then run the command in cmd: pip install tensorflow keras numpy mnist matplotlib
# I am using the tensor virtual environmentn shown in Tech With Tim's video on machine learning

# Import the packages
import numpy as np # To help support large arrays
import mnist # Get data set from
import matplotlib.pyplot as plt # Graph
from keras.models import Sequential # ANN architecture
from keras.layers import Dense # The layers in the ANN
from keras.utils import to_categorical
import keras_preprocessing

# Load the dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalise the images
# Normalise the pixel values from [0, 255] to [-0.5, 0.5] to make network easier to train
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

# Flatten the images, each 28x28 image into a 28^2 = 784 dimenstional vector to pass into the neural network
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Print the shape
print(train_images.shape)
print(test_images.shape)

# Build the model (neural network)
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layer with 10 eurons and softmax function
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model - the loss function measures how well the model did on training, then tries to improve
# it using the optimiser
model.compile(
    optimizer='adam',
    loss= 'categorical_crossentropy', # (Classes that are greater than 2)
    metrics= ['accuracy']
)

# Train the model
model.fit(
    train_images,
    to_categorical(train_labels), # Ex. (returns) 2 it expects [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    epochs = 5, # The number of iterations over the enitre dataset to train on
    batch_size = 32 # The number of samples per gradient update for training
)

# Evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)
# model.save_weights('model.h5') # If we want to save the model

# predict the first 5 test images
predictions = model.predict(test_images[:5])
# Print model's predictions
print(np.argmax(predictions, axis = 1))
# Print the labels (so we know whether it was right or not)
# Should be something like:
# [7 2 1 0 4]
# [7 2 1 0 4]
print(test_labels[:5])

# See the images
for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels)
    # plt.imshow(pixels, cmap='gray') # If we wanted the plots to be black and white
    plt.show()