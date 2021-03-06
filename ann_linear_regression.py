import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 1000
X = np.random.random((N, 2)) * 6 - 3 #creates random butuniformly distributed data between -3 and 3
Y = np.cos(2*X[:,0]) + np.cos(3*X[:,1])

#Plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

#Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
  tf.keras.layers.Dense(1)
])

#compile and fit
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mse')
r = model.fit(X, Y, epochs=100)

plt.plot(r.history['loss'], label='loss')

#plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

#surface plot
line = np.linspace(-3,3,50) #creates 50 evenly spaced points between -3 and 3
xx, yy = np.meshgrid(line,line) #does the cross product between these two lines and assigns it to xx, yy
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T #transforms the data into Nx2 which then stacks these values vertically and transposes them
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()

"""For extrapolation"""

#plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

#surface plot
line = np.linspace(-5,5,50) #creates 50 evenly spaced points between -3 and 3
xx, yy = np.meshgrid(line,line) #does the cross product between these two lines and assigns it to xx, yy
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T #transforms the data into Nx2 which then stacks these values vertically and transposes them
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()
#does not work because did not use periodic activation function
