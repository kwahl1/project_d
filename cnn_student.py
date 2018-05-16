

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

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
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
  

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]


  # -----------------------------------------------------------------------------

  dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
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
    print(mystep)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  # TRAINING ------------------------------------------------------

  labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  P_student = softmax_temperature(logits_student,temperature)
  
   # Calculate Loss (for both TRAIN and EVAL modes)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if distillation == True:
        P_teacher = params["probabilities_teacher"]

        labels_shape = labels_onehot.get_shape()
        N = labels_shape[0].value # size of current subset
        #current_step = tf.get_variable("global_step")
        current_step = 0; tf.train.get_global_step(graph = tf.get_default_graph())

   

        #mystep.assign(mystep+1)
        #print("STEP VALUE")
        #print(mystep.value)
        #time.sleep(5)
     

        #index_min = current_step*N
        #index_max = index_min+N
        
        #print(index_min)
        #print(index_max)

        #P_teacher = P_teacher[index_min:index_max,:]
        P_teacher = tf.convert_to_tensor(P_teacher, dtype=tf.float32)
        #print(P_teacher.get_shape())
        #print(P_student.get_shape())
        print(N)

        #assert 1==2

        P_student = softmax_temperature(logits_student,temperature)
        labels_onehot = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

        # maybe use lambda here
        loss = cross_entropy(labels_onehot,P_student)+cross_entropy(P_teacher,P_student)
    else: 
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_student)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08)
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

  probabilities_teacher = getTargets('soft_targets_teacher')

  #probabilities_teacher = tf.convert_to_tensor(probabilities_teacher) # tensorflow format
  #print(probabilities_teacher.get_shape())

  train_data = train_data[0:1000,:]
  train_labels = train_labels[0:1000]
  probabilities_teacher = probabilities_teacher[0:1000,:]
  #assert 1==2
  print(probabilities_teacher.shape)
  print(train_data.shape)
  print(train_labels.shape)
  #assert 1==2
  
  my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = None,
    save_checkpoints_steps = None
	)

  train_params = {"temperature": 1, "distillation": False, "probabilities_teacher": probabilities_teacher}
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, params = train_params)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)




  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1000,
      num_epochs=None,
      shuffle=False)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps= 500,
      hooks=[logging_hook])






  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)





def getTargets(filename):
	data = np.load(filename+".npy")
	return data


if __name__ == "__main__":
  tf.app.run()

