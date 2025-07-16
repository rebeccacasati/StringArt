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

    n_pegs = 200#256
    image_size = 256#729
    # Side of superpixel cell
    s = 4

    # Target image:
    y_vector = image_to_vector("topolino.jpg", target_size=(image_size, image_size))

    y = y_vector.reshape((image_size, image_size))
    plt.imshow(y, cmap='gray', vmin=0, vmax=1)
    plt.show()

    c = Canvas(image_size=image_size, n_pegs=n_pegs, y_true=y_vector, s=s)

    #x_tilde = np.ones(shape=(len(list(combinations(range(n_pegs), 2))), ), dtype=np.float32)

    #print(x)
    #print(c.possible_threads_combinations)
    #A = c.matrix_A()
    #print(A)
    #c.visualize_A()

    #R, C = c.matrix_D()
    #y = (A @ x_tilde).reshape((image_size*s,image_size*s))
    #y = R @ y @ C

    #plt.matshow(y, cmap='gray_r', vmin=0, vmax=1)
    #plt.show()

    y_pred = c.greedy_optimisation().reshape((image_size,image_size))

    plt.matshow(y_pred, cmap='gray_r', vmin=0, vmax=1)
    plt.savefig(f'Results/Topolino_{n_pegs}_{image_size}.pdf')
    plt.show()

    #y = y.reshape((image_size * image_size * s *s, ))
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
