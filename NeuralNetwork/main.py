import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#load mnist data
mnist = tf.keras.datasets.mnist

#train test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize pixel values for better learning
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#Build a neural network
model = tf.keras.models.Sequential()
#input layer - flatten
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#create hidden layers 128 neurons each, relu activation
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
#create output layer, 10 neurons for 10 digits
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
#compile the model with ADAM optimizer and categorical cross entropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model for 3 epochs
model.fit(x_train, y_train, epochs=3)
#evaluate the model on test data
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.keras')

for x in range(1,5):
    img  = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The number probably is : {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

# def create_digit_image(digit, filename):
#     img = np.ones((28, 28), dtype=np.uint8) * 255  # White background
#     font = cv.FONT_HERSHEY_SIMPLEX
#     cv.putText(img, str(digit), (5, 22), font, 0.8, (0,), 2, cv.LINE_AA)
#     cv.imwrite(filename, img)
#
# # Create four digit images
# digits = [3, 6, 8, 9]
#
# for digit in digits:
#     filename = f"{digit}.png"
#     create_digit_image(digit, filename)