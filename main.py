import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from scipy.sparse import lil_matrix
from canvas import Canvas
from utils import image_to_vector
from itertools import combinations
from scipy.sparse.linalg import lsqr




def main():

    n_pegs = 100#256
    image_size = 128#729
    # Side of superpixel cell
    s = 4

    # Target image:
    y_vector = image_to_vector("Taylor_Swift.jpg", target_size=(image_size, image_size))

    y = y_vector.reshape((image_size, image_size))
    plt.imshow(y, cmap='gray', vmin=0, vmax=1)
    plt.show()


    c = Canvas(image_size=image_size, n_pegs=n_pegs, y_true=y_vector, s=s)

    H = c.matrix_H()

    y = H @ y
    plt.imshow(y, cmap='gray', vmin=0, vmax=1)
    plt.show()

    #print('Pixels: ', c.pixels)
    #print('Pegs: ', c.anchor_pegs)
    #x_tilde = np.ones(shape=(len(list(combinations(range(n_pegs), 2))), ), dtype=np.float64)

    #print(x)
    #print(c.possible_threads_combinations)
    #A = c.matrix_A()
    #print(A)
    #c.visualize_A()

    #y = (A @ x_tilde).reshape((image_size*s,image_size*s))


    sys.exit()

    y_pred = c.greedy_optimisation()

    plt.matshow(y_pred, cmap='gray', vmin=0, vmax=1)
    plt.show()

    #sys.exit()
    #y = y.reshape((image_size * image_size * s *s, ))

    # initialization of edge vector
    #x = np.array(...)

    # compute approximated image
    #y_pred = A @ x
    # clamping
    #CAy_pred = c.matrix_C(y)
    # dimensionality reduction
    #DCAy_pred = c.matrix_D(CAy_pred)
    #print(DCA.shape)

    #sys.exit()

    # linear leas squares optimizer
    #result = lsqr(DCA, y)
    #x = result[0]  # Solution vector
    #print("Solution x:", x)
    #print("Residual norm ||Ax - y||:", result[3])  # result[3] is the norm of residual

    #y_pred = DCA @ x
    #plt.hist(y_pred, bins=20)
    #plt.show()
    #y_pred = y_pred.reshape((image_size, image_size))
    #plt.matshow(y_pred, cmap='gray', vmin=0, vmax=1)
    #plt.savefig(f'Results/Einstein_{n_pegs}_{image_size}.pdf')
    #plt.show()


if __name__ == '__main__':
    main()