import numpy as np
from ellipsoid_fit import ellipsoid_fit as ellipsoid_fit, data_regularize
import matplotlib.pyplot as plt


if __name__ == '__main__':

    data = np.loadtxt("mag_out.txt")
    # data2 = data_regularize(data)

    center, evecs, radii, v = ellipsoid_fit(data)

    #a, b, c = radii
    #r = (a * b * c) ** (1. / 3.)
    #D = np.array([[r/a, 0., 0.], [0., r/b, 0.], [0., 0., r/c]])
    #transformation = evecs.dot(D).dot(evecs.T)

    print('')
    print('center: ', center)
    print('radii: ', radii)
    print('evecs: ', evecs)
    #print('transformation:')
    #print(transformation)

    #np.savetxt('magcal_ellipsoid.txt', np.vstack((center.T, transformation)))

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    X = np.array([x * x,
                  y * y,
                  z * z,
                  2 * x * y,
                  2 * x * z,
                  2 * y * z,
                  2 * x,
                  2 * y,
                  2 * z,
                  1 - 0 * x])
    X = X.T
    loss = []
    
    for p in X:
        loss.append(pow(np.dot((v.T), p), 1))
    
    for i in range(len(loss)):
        plt.scatter(i, loss[i], marker='.')