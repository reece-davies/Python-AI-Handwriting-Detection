# Description: Classifies the MNIST handwritten digit images as number 0-9
# https://www.youtube.com/watch?v=kOFUQB7u5Ck&list=WL&index=6&t=301s

# Then use the guide for making a drawing application so it can predict what you have drawn
# https://www.youtube.com/watch?v=WVuMCawju-g&list=PLujzF9oKZSaTU_nEN9cRjMeEdhcDQT7sh&index=77&t=86s

# Additionally, we could make our own images using gimp and then run the ANN to predict what number they are:
# https://www.youtube.com/watch?v=E6CxY8gspfQ


# Download pip (or Anaconda), and then run the command in cmd: pip install tensorflow keras numpy mnist matplotlib
# I am using the tensor virtual environmentn shown in Tech With Tim's video on machine learning

#############
# ANN setup #
#############

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

#################################################
first_image = test_images[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
#plt.imshow(pixels, cmap='gray') # If we wanted the plots to be black and white
plt.imshow(pixels)
plt.show()
print("Image has size: ", first_image.size)
#################################################

# Normalise the images
# Normalise the pixel values from [0, 255] to [-0.5, 0.5] to make network easier to train
#train_images = (train_images/255) - 0.5
#test_images = (test_images/255) - 0.5
train_images = np.array(train_images)/255.0 - 0.5
test_images = np.array(test_images)/255.0 - 0.5

# Flatten the images, each 28x28 image into a 28^2 = 784 dimenstional vector to pass into the neural network
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Print the shape
print(train_images.shape)
print(test_images.shape)

print("Test_images[0] shape: ")
print(test_images[0].shape)

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

# Evaluate the model#############################################################
model.evaluate(
    test_images,
    to_categorical(test_labels)
)
# model.save_weights('model.h5') # If we want to save the model
print("Print 1st 5 test_images")
print(test_images[:5])

# predict the first 5 test images
predictions = model.predict(test_images[:5])###############################
# Print model's predictions
print(np.argmax(predictions, axis = 1))########################################
print(predictions)
# Print the labels (so we know whether it was right or not)
# Should be something like:
# [7 2 1 0 4]
# [7 2 1 0 4]
print(test_labels[:5])#######################################

# See the images
#for i in range(0, 5):
#    first_image = test_images[i]
#    first_image = np.array(first_image, dtype='float')
#    pixels = first_image.reshape((28, 28))
#    # plt.imshow(pixels) # Original colour
#    plt.imshow(pixels, cmap='gray') # If we wanted the plots to be black and white
#    plt.show()



##############################
# Drawing app implementation #
##############################
import pygame
from PIL import Image
import tensorflow as tf

import ctypes  # An included library with Python install. Popup message box 
 
pygame.init()
screen = pygame.display.set_mode((560,560))
pygame.display.set_caption("Artificial Neural Network Digit Predictor")
clock = pygame.time.Clock()
 
loop = True

while loop:
    try:
        #pygame.mouse.set_visible(False)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loop = False
            elif event.type == pygame.KEYDOWN:
                # Save the screen as an image
                fileName = "image.png"
                pygame.image.save(screen, fileName)
                print("File {} has been saved".format(fileName))

                # Load the image into the screen
                printImage = pygame.image.load(fileName)
                screen.blit(printImage, (5,5)) # (0,0)
                print("Loaded {} onto screen".format(fileName))

                #############################################################################

                print("Image shape")
                IMAGE_SHAPE = (28, 28)
                print("Get file testImage (taken out)")
                #testImage = tf.keras.utils.get_file('image.jpg',"C:\\Users\\Reece Davies")
                print("Open testImage")
                testImage = Image.open(fileName).resize(IMAGE_SHAPE)

                # Convert modules: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
                # Converts image to 8-bit pixels, mapped to any other mode using a color palette
                testImage = testImage.convert('P') # The one that works best

                testImage = np.array(testImage, dtype='float') # I don't know whether this makes an improvement or not

                #############################################################################

                display_image = testImage
                display_image = np.array(display_image, dtype='float')
                #plt.imshow(display_image, cmap='gray') # If we wanted the plots to be black and white
                plt.imshow(display_image)
                plt.show()
                print("Plot image has size: ", testImage.size)

                print("Image has size: ", testImage.size)

                print("Normalising image ")

                #-------------------------------- # Guide for how to normalise images in python: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/

                print("Normalizing image using online guide")
                from numpy import asarray
                testImage = asarray(testImage)
                # confirm pixel range is 0-255
                print('Data Type: %s' % testImage.dtype)
                print('Min: %.3f, Max: %.3f' % (testImage.min(), testImage.max()))
                # convert from integers to floats
                testImage = testImage.astype('float32')
                # normalize to the range 0-1
                #testImage /= 255.0
                testImage = np.array(testImage)/255.0 - 0.5 # MAIN ONE
                # confirm the normalization
                print('Min: %.3f, Max: %.3f' % (testImage.min(), testImage.max()))
                

                #--------------------------------
                print("Shape is: ")
                print("Shape is: ", testImage.shape)        # I need to change the reshaping of the image, so the size is 784 (28*28 = (,784)) and not 2352 (28*28*3 = (28,28,3))
                print("Image has size: ", testImage.size)   # Use this guide to changing image size (Based in @tomvon): https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio

                # Flatten the images, each 28x28 image into a 28^2 = 784 dimenstional vector to pass into the neural network
                print("Flattening image ")
                testImage = testImage.reshape((-1, 784))

                print("Image has size: ", testImage.size)

                # Print the shape
                print("Displaying image shape ")
                print(testImage.shape)

                print(testImage)

                # predict the first 5 test images
                print("Predicting image")
                #inputPrediction = model.predict(testImage)
                inputPrediction = model.predict(testImage)

                # Print model's predictions
                print("Displaying input prediction: ")
                print(np.argmax(inputPrediction, axis = 1)) # Prints weird array of 3 digits, not 1 digit
                print(inputPrediction)


                #############################################################################

                # Inform the user of number predicted and then fill the screen
                message = "Number predicted: {}".format(np.argmax(inputPrediction, axis = 1)) 
                ctypes.windll.user32.MessageBoxW(0, message, "Your title", 1)
                screen.fill((0, 0, 0))
                    

        px, py = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1,0,0):
            pygame.draw.rect(screen, (255,255,255), (px-15,py-15,30,30))

            
        pygame.display.update()

        clock.tick(1000)
    except Exception as e:
        print(e)
        #pygame.quit()
        exit()
        


pygame.quit()
