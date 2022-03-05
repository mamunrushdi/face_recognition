# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:18:27 2022

@author: mamun


Face Recognition
 
Face recognition problems commonly fall into one of two categories:

Face Verification "Is this the claimed person?" For example, at some airports, 
you can pass through customs by letting a system scan your passport and then verifying that you 
(the person carrying the passport) are the correct person. A mobile phone that unlocks using your 
face is also using face verification. This is a 1:1 matching problem.

 
FaceNet learns a neural network that encodes a face image into a vector of 128 numbers. 
By comparing two such vectors, you can then determine if two pictures are of the same person.

In this assignment, we'll do:

# Differentiate between face recognition and face verification
# Implement one-shot learning to solve a face recognition problem
# Apply the triplet loss function to learn a network's parameters in the context of face recognition
# Explain how to pose face recognition as a binary classification problem
# Map face images into 128-dimensional encodings using a pretrained model
# Perform face verification and face recognition with these encodings
# Channels-last notation

For this assignment, we'll be using a pre-trained model which represents ConvNet activations using a 
"channels last" convention, as used during the lecture and in previous programming assignments.

In other words, a batch of images will be of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶) .

"""

#%% import packages and utilites
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL

print("packages and utilites import completed")

#%% Naive Face Verification
'''
Encoding Face Images into a 128-Dimensional Vector

3.1 - Using a ConvNet to Compute Encodings
The FaceNet model takes a lot of data and a long time to train. So following the common practice in applied deep learning, 
we'll load weights that someone else has already trained. The network architecture follows the Inception model from 
Szegedy et al.. An Inception network implementation has been provided for you, and you can find it in the file 
inception_blocks_v2.py to get a closer look at how it is implemented.

Hot tip: Go to "File->Open..." at the top of this notebook. This opens the file directory that contains the .py file).

The key things we have to be aware of are:

This network uses 160x160 dimensional RGB images as its input. 
Specifically, a face image (or batch of  𝑚  face images) as a tensor of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3) 
The input images are originally of shape 96x96, thus, you need to scale them to 160x160. 
This is done in the img_to_encoding() function.
The output is a matrix of shape  (𝑚,128)  that encodes each input face image into a 128-dimensional vector
Run the cell below to create the model for face images!

'''
#%%
from tensorflow.keras.models import model_from_json

json_file = open('keras-facenet-h5/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/keras-facenet-h5/weights.h5')


#%%
print(model.inputs)
print(model.outputs)

#%%
# example of loading the keras facenet model
from keras.models import load_model

#%% load the model
facenet_model = load_model('D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/keras-facenet-h5/facenet_keras.h5')

#%% summarize input and output shape
print(facenet_model.inputs)
print(facenet_model.outputs)

#%%
'''
By using a 128-neuron fully connected layer as its last layer, 
the model ensures that the output is an encoding vector of size 128. 

So, an encoding is a good one if:

The encodings of two images of the same person are quite similar to each other.
The encodings of two images of different persons are very different.
The triplet loss function formalizes this, and tries to "push" the encodings of two images of the same person (Anchor and Positive) closer together, while "pulling" the encodings of two images of different persons (Anchor, Negative) further apart
'''

#%% The Triplet Loss

'''
The Triplet Loss

**Important Note**: 
    the triplet loss is the main ingredient of the face recognition algorithm, 
    and you'll need to know how to use it for training your own FaceNet model, 
    as well as other types of image similarity problems. 
    Therefore, you'll implement it below, for fun and edification. :) 
'''
#%%
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
   
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

#%% test triplet_loss
tf.random.set_seed(1)
y_true = (None, None, None) # It is not used
y_pred = (tf.keras.backend.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=1, stddev=1, seed = 1),
          tf.keras.backend.random_normal([3, 128], mean=3, stddev=4, seed = 1))
loss = triplet_loss(y_true, y_pred)

# assert type(loss) == tf.python.framework.ops.EagerTensor, "Use tensorflow functions"
# print("loss = " + str(loss))

y_pred_perfect = ([1., 1.], [1., 1.], [1., 1.,])
loss = triplet_loss(y_true, y_pred_perfect, 5)
assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
y_pred_perfect = ([1., 1.],[1., 1.], [0., 0.,])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
y_pred_perfect = ([1., 1.],[0., 0.], [1., 1.,])
loss = triplet_loss(y_true, y_pred_perfect, 0)
assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
y_pred_perfect = ([0., 0.],[0., 0.], [0., 0.,])
loss = triplet_loss(y_true, y_pred_perfect, -2)
assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
loss = triplet_loss(y_true, y_pred_perfect, 3)
assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"

#%% Loading the Pre-trained Model
'''
FaceNet is trained by minimizing the triplet loss. But since training requires a lot of data 
and a lot of computation, we won't train it from scratch here. Instead, we'll load a previously 
trained model in the following cell; which might take a couple of minutes to run.

'''
FRmodel = model

#%% Applying the Model
'''
We're building a system for an office building where the building manager would like to offer facial 
recognition to allow the employees to enter the building.

We'd like to build a face verification system that gives access to a list of people. 
To be admitted, each person has to swipe an identification card at the entrance. 
The face recognition system then verifies that they are who they claim to be
'''

#%% Face Verification
'''
Now we'll build a database containing one encoding vector for each person who is allowed to enter the office. 
To generate the encoding, we'll use img_to_encoding(image_path, model), 
which runs the forward propagation of the model on the specified image.

Run the following code to build the database (represented as a Python dictionary). 
This database maps each person's name to a 128-dimensional encoding of their face.

'''

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

#%% test img_to_encoding
database = {}
database["danielle"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/arnaud.jpg", FRmodel)


#%% Load the images of Danielle and Kian:
danielle = tf.keras.preprocessing.image.load_img("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/danielle.png", 
                                                 target_size=(160, 160))
kian = tf.keras.preprocessing.image.load_img("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/kian.jpg", 
                                             target_size=(160, 160))
#%%
np.around(np.array(kian) / 255.0, decimals=12).shape

#%% print images
kian

#%%
np.around(np.array(danielle) / 255.0, decimals=12).shape
#%% 
danielle

#%% verify
'''

Implement the verify() function, which checks if the front-door camera picture (image_path) 
is actually the person called "identity". You will have to go through the following steps:

Compute the encoding of the image from image_path.
Compute the distance between this encoding and the encoding of the identity image stored in the database.
Open the door if the distance is less than 0.7, else do not open it.
As presented above, you should use the L2 distance np.linalg.norm.

Note: In this implementation, compare the L2 distance, not the square of the L2 distance, to the threshold 0.7.

Hints:

identity is a string that is also a key in the database dictionary.
img_to_encoding has two parameters: the image_path and model
'''

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras
    
    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
     
    return dist, door_open

#%% 
assert(np.allclose(verify("images/camera_1.jpg", "bertrand", database, FRmodel), (0.54364836, True)))
assert(np.allclose(verify("images/camera_3.jpg", "bertrand", database, FRmodel), (0.38616243, True)))
assert(np.allclose(verify("images/camera_1.jpg", "younes", database, FRmodel), (1.3963861, False)))
assert(np.allclose(verify("images/camera_3.jpg", "younes", database, FRmodel), (1.3872949, False)))
# verify("images/camera_0.jpg", "younes", database, FRmodel)

#%%
verify("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/camera_2.jpg", "kian", database, FRmodel)

#%% Face Recognition
'''
face verification system is mostly working. 
But since Kian got his ID card stolen, when he came back to the office the next day he couldn't get in!

To solve this, we'd like to change your face verification system to a face recognition system. 
This way, no one has to carry an ID card anymore. 
An authorized person can just walk up to the building, and the door will unlock for them!

We'll implement a face recognition system that takes as input an image, 
and figures out if it is one of the authorized persons (and if so, who). 
Unlike the previous face verification system, we will no longer get a person's name as one of the inputs.

'''

#%%  who_is_it
'''

Implement who_is_it() with the following steps:

Compute the target encoding of the image from image_path
Find the encoding from the database that has smallest distance with the target encoding.
Initialize the min_dist variable to a large enough number (100). 
This helps you keep track of the closest encoding to the input's encoding.
Loop over the database dictionary's names and encodings. 
To loop use for (name, db_enc) in database.items().
Compute the L2 distance between the target "encoding" and the current "encoding" from the database. 
If this distance is less than the min_dist, then set min_dist to dist, and identity to name.

'''
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras
    
    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """
   
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - database[name])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

#%% # Test 1 with Younes pictures 
who_is_it("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/camera_0.jpg", database, FRmodel)

# Test 2 with Younes pictures 
test1 = who_is_it("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/camera_0.jpg", database, FRmodel)
assert np.isclose(test1[0], 0.5992946)
assert test1[1] == 'younes'

# Test 3 with Younes pictures 
test2 = who_is_it("D:/deep_learning_lab/deep_learning/convolutional_neural_network/Week4/Face Recognition/images/younes.jpg", database, FRmodel)
assert np.isclose(test2[0], 0.0)
assert test2[1] == 'younes'
