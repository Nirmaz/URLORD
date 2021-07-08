import numpy as np
import matplotlib.pyplot as plt





















if __name__ == '__main__':

    a = np.zeros((2,2))
    a[0:2,0] = 0.7
    a[0:2,1] = -0.8
    plt.imshow(a, cmap='gray')
    plt.show()