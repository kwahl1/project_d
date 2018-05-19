

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode, params ):

  temperature = params["temperature"]
  distillation = params["distillation"]
  
  #labels = tf.convert_to_tensor(labels[:,0], dtype=tf.int32)
  P_teacher = labels[:,1:11]
  labels = labels[:,0]
  labels = tf.cast(labels,tf.int32)

  print(type(labels))
  print(labels.shape)
  print(labels)


  

  #global_step = tf.Variable(0, name='global_step', trainable=False)

  #logits_teacher = params["logits_teacher"]
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]



  '''
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 16])
  '''



  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  '''
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=16,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
  '''

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
 
  '''
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  #pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 128])

  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  pool4_flat = tf.reshape(pool4, [-1, 1 * 1 * 128])
  '''

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]


  # -----------------------------------------------------------------------------
  #units = 1024
  input_flat = tf.reshape(input_layer, [-1, 28*28])
  dense = tf.layers.dense(inputs=input_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  #0.4
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  
  logits_student = tf.layers.dense(inputs=dropout, units=10)


  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits_student, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits_student, name="softmax_tensor")
  }

  # PREDICTION ----------------------------------------------------

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # TRAINING ------------------------------------------------------

  labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  P_student = softmax_temperature(logits_student,temperature)
  
   # Calculate Loss (for both TRAIN and EVAL modes)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:

    if distillation == True:

        labels_shape = labels_onehot.get_shape()
        N = labels_shape[0].value # size of current subset

        P_teacher = tf.cast(P_teacher,tf.float32) 

        P_student = softmax_temperature(logits_student)
        P_student_temperature = softmax_temperature(logits_student,temperature)
        #labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

        # maybe use lambda here
        T_square = 1#np.square(temperature)
        lambda_ = 1
        #assert 1==2

        #loss = T_square*cross_entropy(labels_onehot,P_student)+lambda_*T_square*cross_entropy(P_teacher,P_student_temperature)
        loss_hard = T_square*cross_entropy_2(labels_onehot,P_student)
        loss_soft = T_square*cross_entropy_2(P_teacher,P_student_temperature)

        loss = loss_hard+lambda_*loss_soft

        print(type(loss))
        print(T_square)
        print(tf.shape(loss_hard))
        print(tf.shape(loss_soft))
        print(tf.shape(loss_hard+loss_soft))
        print(tf.shape(loss))
        #assert 1==2
    else: 
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_student)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



  # EVALUATE ------------------------------------------------------------------------
  
  loss_eval = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_student)
 
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss_eval, eval_metric_ops=eval_metric_ops, predictions = predictions)



def cross_entropy_2(labels_onehot,probability):
  #y_transpose = tf.transpose(labels_onehot)
  tensor_shape = labels_onehot.get_shape()
  N = tensor_shape[0].value
  assert not N==None
  assert not N==0
  p_transpose = tf.transpose(probability)


  l_cross = -1*tf.diag_part(tf.matmul(labels_onehot,tf.log(p_transpose)))
  return tf.divide(tf.reduce_sum(l_cross),N)


def softmax_temperature(logits, T=1):
    logits_temp = tf.divide(logits,T)
    softmax_temp = tf.divide(tf.exp(logits_temp),
    tf.reduce_sum(tf.exp(logits_temp),axis=1, keep_dims=True))
    return softmax_temp



def cross_entropy(labels_onehot,probability):
    # incorrect entropy, use cross_entropy_2
    
    tensor_shape = labels_onehot.get_shape()
    N = tensor_shape[0].value
    assert not N==None
    assert not N==0
    p_transpose = tf.transpose(probability)
    l_cross = -1*tf.log(tf.diag_part(tf.matmul(labels_onehot,p_transpose)))
    return tf.divide(tf.reduce_sum(l_cross),N)



def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  probabilities_teacher = readFile('soft_targets_teacher')
  train_labels_both = np.column_stack([train_labels,probabilities_teacher])
  eval_labels = np.transpose(np.matrix(eval_labels))



  
  my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = None,
    save_checkpoints_steps = None
	)

  train_params = {"temperature": 2, "distillation": True, "probabilities_teacher": probabilities_teacher}
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, params = train_params)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1000)




  # Train the model

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels_both,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps= 1000,
      hooks=[logging_hook])





  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)





def readFile(filename):
	data = np.load(filename+".npy")
	return data


if __name__ == "__main__":
  tf.app.run()

