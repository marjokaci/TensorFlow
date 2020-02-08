#gitchange

# Linear Regression Using Tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# In order to make the random numbers predictable, we will define fixed seeds for both Numpy and Tensorflow.
np.random.seed(101)
tf.compat.v1.set_random_seed(101)

# Genrating random linear data
# There will be 50 data points ranging from 0 to 50
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)


# Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x) # Number of data points


# Plot of Training Data
plt.scatter(x, y)
plt.xlabel('x')
plt.xlabel('y')
plt.title("Training Data")
plt.show()


# Now we will start creating our model by defining the placeholders X and Y, so that we can feed our training examples
# X and Y into the optimizer during the training process.
X = tf.compat.v1.placeholder("float")
Y = tf.compat.v1.placeholder("float")



# Now we will declare two trainable Tensorflow Variables for the Weights and Bias and initializing them randomly
W = tf.Variable(np.random.randn(), name = "W")
b = tf.Variable(np.random.randn(), name = "b")


# Now we will define the hyperparameters of the model, the Learning Rate and the number of Epochs.

learning_rate = 0.01
training_epochs = 1000


# Hypothesis
y_pred = tf.add(tf.multiply(X, W), b)

# Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

# Gradient Descent Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.compat.v1.global_variables_initializer()


# Starting the Tensorflow Session
with tf.compat.v1.Session() as sess:

    # Initializing the Variables
    sess.run(init)

    # Iterating through all the epochs
    for epoch in range(training_epochs):

        # Feeding each data point into the optimizer using Feed Dictionary
        for (_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict = {X : _x, Y : _y})

            # Displaying the result after every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Calculating the cost a every epoch
            c = sess.run(cost, feed_dict = {X : x, Y : y})
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))

            # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)


# Calculating the predictions
predictions = weight * x + bias
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')
