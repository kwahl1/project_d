
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import imageio
import glob, os
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)




  def cnn_model_fn_student(features, labels, mode):

  temperature = params["temperature"]
  distillation = params["distillation"]

  P_teacher = labels[:,1:11]
  P_teacher = tf.cast(P_teacher,tf.float32)
  labels = labels[:,0]
  labels = tf.cast(labels,tf.int32)


  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]  # [batch_size, 64, 64, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32] # [batch_size, 64, 64, 1]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32] # [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]  # [batch_size, 32, 32, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32] # [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64] #[batch_size, 32, 32, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]  #[batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 7, 7, 64]     #[batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]  #[batch_size, 16, 16, 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 128]) # 8*8*64
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense1 = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)

  dense2 = tf.layers.dense(inputs=dense1, units=2048, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10] #[batch_size, 358]
  logits_student = tf.layers.dense(inputs=dropout, units=358)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits_student, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits_student, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)




  # Calculate Loss (for both TRAIN and EVAL modes)

  labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  P_student = softmax_temperature(logits_student,temperature)
  
   # Calculate Loss (for both TRAIN and EVAL modes)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:

    if distillation == True:

        labels_shape = labels_onehot.get_shape()
        N = labels_shape[0].value # size of current subset

         

        P_student = softmax_temperature(logits_student)
        P_student_temperature = softmax_temperature(logits_student,temperature)
        #labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

        # maybe use lambda here
        T_square = 1 #np.square(temperature)
        lambda_ = 1

        loss = T_square*cross_entropy(labels_onehot,P_student)+lambda_*T_square*cross_entropy(P_teacher,P_student_temperature)
    else: 
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_student)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)





  loss_eval = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_student)


  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss_eval, eval_metric_ops=eval_metric_ops)






def softmax_temperature(logits, T=1):
    logits_temp = tf.divide(logits,T)
    softmax_temp = tf.divide(tf.exp(logits_temp),
      tf.reduce_sum(tf.exp(logits_temp),axis=1, keep_dims=True))
    return softmax_temp



def cross_entropy(labels_onehot,probability):
    #y_transpose = tf.transpose(labels_onehot)
    tensor_shape = labels_onehot.get_shape()
    N = tensor_shape[0].value
    assert not N==None
    assert not N==0
    p_transpose = tf.transpose(probability)
    l_cross = -1*tf.log(tf.diag_part(tf.matmul(labels_onehot,p_transpose)))
    return tf.divide(tf.reduce_sum(l_cross),N)

def felixHALLÅ():
  probabilities_teacher = readFile('soft_targets_teacher')
  train_labels_both = np.column_stack([train_labels,probabilities_teacher])
  eval_labels = np.transpose(np.matrix(eval_labels))
  return


def trainModel(trainSamples,trainLabels,testSamples,testLabels):
  # Load training and eval data


  probabilities_teacher = readFile('soft_targets_teacher')
  train_labels_both = np.column_stack([trainLabels,probabilities_teacher])
  testLabels = np.transpose(np.matrix(testLabels))
  

  [train_data, train_labels] = randomize(trainSamples,trainLabels)
  [eval_data, eval_labels] = randomize(testSamples,testLabels)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn,params = {"temperature": 2, "distillation": True}, model_dir="/tmp/mnist_convnet_model5")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=100,
      )

  # Evaluate the model and print results
  #evalRes = evalModel(eval_data, eval_labels)
  #print(evalRes)
  evalRes = evalModel(testSamples, testLabels)
  print(evalRes)

#--------------Eval Model------------
#---------------------------------

#Returns a matrix with the probabilities for all classess
def predModel(testData):
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model5")

    eval_data = testData
    
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        num_epochs=1,
        shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input_fn)
    resultsList = list(pred_results)
    nImages = len(resultsList)
    nClasses = len(resultsList[0]['probabilities'])
    probabilities = np.empty((nImages,nClasses))
    i = 0
    for res in resultsList:
        probabilities[i,:] = res['probabilities']
        i +=1
    
    return probabilities

#Returns the accuarcy over the test data
def evalModel(testData,testLabels):
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model5")
    
    eval_data = testData
    eval_labels = testLabels
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    return eval_results

def isItFood(testProbability,testLabels):
    nData = len(testLabels)
    nCor = 0
    for i in range(0,nData):
        label = np.argmax(testProbability[i,:])
        if(label < 101 and testLabels[i] < 101):
            nCor +=1
        elif(label >= 101 and testLabels[i] >= 101):
            nCor +=1
    return nCor/nData

def predictSingleIm(subject,image):
    with open("itemList.txt") as f:
        mylist = f.read().splitlines()
    sample = imageio.imread(image)

    sample = np.reshape(sample,(1,64*64))/255
    sample = np.asarray(sample, dtype=np.float32)
    prob = predModel(sample)
    label = np.argmax(prob)
    print("Compiling nodes...")
    print("Subject",subject,"looks like a:",mylist[label])
    
    
    

#-------Data Manipulation------
#Reads images and returns a matrix representatin of the matrices
def readImData(rootdir,Nimages,imRes):
    i = 0
    j = 0;
    images = np.empty((Nimages,imRes))
    labels = np.empty((Nimages))
    print("Loading",Nimages,"images...")
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            
            im = imageio.imread(os.path.join(subdir, file))
            images[i,:] = np.reshape(im,(1,imRes))/255
            
            labels[i] = j
            i = i+1
        j = j +1
        print(j)
    labels = labels -1
    labels = np.asarray(labels, dtype=np.int32)
    images = np.asarray(images, dtype=np.float32)
    print("Loaded",Nimages,"images")
    return images, labels
    
def countImages(rootdir):
    nImages = 0
    for path, subdirs, files in os.walk(rootdir):
        nImages = nImages + len(files)
    return nImages

def randomize(data,labels):
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels

#Saves a matrix(data) to directory(directory) with name fileName.npy
#directory: str, fileName, str
def saveData(directory,data,fileName):
    filePath = os.path.join(directory,fileName+".npy")
    np.save(filePath,data)
    
#directory, where the file is. fileName, name of the file without extension
def loadData(directory,fileName):
    filePath = os.path.join(directory,fileName+".npy")
    if(os.path.isfile(filePath)):
        print("Loaded:",fileName)
        return np.load(filePath)
    else:
        print("No such file exists:",fileName)

trainSamples = loadData("C:/Users/Felix/Desktop/DL FinalProject/DATA","trainSamples")
trainLabels = loadData("C:/Users/Felix/Desktop/DL FinalProject/DATA","trainLabels")
testSamples = loadData("C:/Users/Felix/Desktop/DL FinalProject/DATA","testSamples")
testLabels = loadData("C:/Users/Felix/Desktop/DL FinalProject/DATA","testLabels")
#pred=predModel(testSamples)
#acc1 = isItFood(pred,testLabels)
#acc2 = evalModel(testSamples,testLabels)
trainProb = loadData("C:/Users/Felix/Desktop/DL FinalProject/DATA","trainProb")
trainModel(trainSamples,trainProb,testSamples,testLabels)
#evalModel(testSamples,testLabels)
#predictSingleIm("Krille","krille.png")
#evalModel(trainSamples,trainProb)
#nFood = countImages("C:/Users/Felix/Desktop/DL FinalProject/DATA/FOOD_Grey_Testing")
#nFood2 = countImages("C:/Users/Felix/Desktop/DL FinalProject/DATA/notFOOD_Grey_Testing")
#[images, labels] = readImData("C:/Users/Felix/Desktop/DL FinalProject/DATA/FOOD_Grey_Testing",nFood,64*64)
#[images2, labels2] = readImData("C:/Users/Felix/Desktop/DL FinalProject/DATA/notFOOD_Grey_Testing",nFood2,64*64)

#data = np.vstack((images,images2))
#data2 = np.concatenate((labels, labels2 +101), axis=0)
#saveData("C:/Users/Felix/Desktop/DL FinalProject/DATA",data,"testSamples")
#saveData("C:/Users/Felix/Desktop/DL FinalProject/DATA",data2,"testLabels")







