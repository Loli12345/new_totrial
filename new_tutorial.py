import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from urllib import urlretrieve
import cPickle as pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import gzip
from skimage.feature import canny
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.base import TransformerMixin
#import theano
import csv
from skimage.filters import sobel
import pandas as pd
import sklearn
from itertools import islice
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
###############################################
IMAGE_WIDTH = 176
CHANNELS = 1
IMAGE_HEIGHT=208
n_input=36608
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS
def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
#load modified data
def load_dataset():
    imList = []
    for root, dirs, files in os.walk("/Users/elhamalkabawi/Desktop/disc2"):
        for file in files:
            if file.endswith(".gif"):
                filename = os.path.join(root, file)
                image = Image.open(filename, 'r')
                image= image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
                img=np.array(image)
                new_img, h, new_h, sk = histeq(img)
                elevation_map = sobel(new_img)
                edges = canny(elevation_map /255.)
                b=normalize(edges)
                imList.append(b)


    X_train, X_test =train_test_split(imList, test_size = 0.4)
    X_train = np.array(X_train)
    print(X_train)
    X_test=np.array(X_test)
    print(X_test.shape)
    X_train=X_train.astype(np.float32)
    X_train = X_train.reshape(440,IMAGE_PIXELS)
    X_test = X_test.reshape(294,IMAGE_PIXELS)
    X_test=X_test.astype(np.float32)
 #   X_train = X_train.reshape((-1, 1, IMAGE_WIDTH,IMAGE_HEIGHT))
 #   X_test = X_test.reshape((-1, 1, IMAGE_WIDTH,IMAGE_HEIGHT))
    print(X_train.shape)
    print(X_test.shape)
    #y_train = y_train.astype(np.uint8)
 #   y_test = y_test.astype(np.uint8)
    return X_train, X_test
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
names=['map','count','Age','eTIV','ETA','nWBV','L. HC','R. HC','CDR']
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
X_train, X_test= load_dataset()    
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2
X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
encoder_op = encoder(X_train)
print(encoder_op )
decoder_op = decoder(encoder_op)
print(encoder_op )
# Prediction
y_pred = decoder_op
print('pred')
print(y_pred)
encoder_op1 = encoder(X_test)
print(encoder_op1 )
decoder_opn = decoder(encoder_op1)

# Prediction
compressed_Xtest = decoder_opn
print(compressed_Xtest)
csv = pd.read_csv('/Users/elhamalkabawi/Desktop/research/OASIS_data.csv',names=names)
xt= DataFrameImputer().fit_transform(csv)
array = xt.values
X = array[1:735,0:8]
Y = array[1:735,8]
label=[]
for i in Y:
    if i=='No dementia':
            label.append(0.0)
    else:
        if i=='Incipient demt PTP':
            #print i
            label.append(1.0)
        else:
           if i=='uncertain dementia':
               label.append(2.0)
           else:
               if i=='DAT':
                   label.append(3.0)
               else:
                   label.append(4.0)
y_train, y_test= sklearn.cross_validation.train_test_split(label, train_size = 0.6)
y_train=np.array(y_train)
y_test=np.array(y_test)
print(y_train)
print(y_test.shape)
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)
init = tf.initialize_all_variables()
#########################
with tf.Session() as sess:
        sess.run(init)
        y_pred=y_pred.eval()
        predicted=y_pred.reshape((-1, 1, IMAGE_WIDTH,IMAGE_HEIGHT))
        print(predicted.dtype)
        compressed_Xtest=compressed_Xtest.eval()
        Xl=compressed_Xtest.reshape((-1, 1, IMAGE_WIDTH,IMAGE_HEIGHT))
   
#plt.imshow(X_train[0][0], cmap=cm.binary)
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 176,208),
    # layer conv2d1
    conv2d1_num_filters=16,
    conv2d1_filter_size=(8,8),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    # layer maxpool1
    maxpool1_pool_size=(2, 2),    
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.GlorotUniform(),  
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d3
    conv2d3_num_filters=64,
    conv2d3_filter_size=(5, 5),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool3
    maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,    
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,    
    # dropout2
    dropout2_p=0.5,    
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=5,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=3,
    verbose=1,
    )
# Train the network
nn = net1.fit(predicted, y_train)
#m=LogisticRegression()
#m.fit(y_pred,y_train)
#preds=m.predict(X_test)
preds = net1.predict(Xl)
print(preds)
print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,preds)*100))
cm = confusion_matrix(y_test, preds)
print(y_test)
print(cm)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
