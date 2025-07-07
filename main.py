import sys
import matplotlib.pyplot as plt
import torch
from skimage.draw import line
from scipy.sparse import lil_matrix
from canvas import Canvas
from utils import image_to_vector
from itertools import combinations
from scipy.sparse.linalg import lsqr



def main():

    n_pegs = 100#256
    image_size = 256#729

    # Target image:
    y = image_to_vector("Albert_Einstein.jpg", target_size=(image_size, image_size))

    y = y.reshape((image_size, image_size))
    plt.imshow(y, cmap='gray', vmin=0, vmax=1)
    plt.show()


    #sys.exit()
    c = Canvas(image_size=image_size, n_pegs=n_pegs)
    #print('Pixels: ', c.pixels)
    #print('Pegs: ', c.anchor_pegs)
    x_tilde = torch.ones(size=(len(list(combinations(range(n_pegs), 2))), ), dtype=torch.float64)

    #print(x)
    #print(c.possible_threads_combinations)
    A = c.matrix_A()
    #c.visualize_A()

    #y = (A @ x_tilde).reshape((image_size,image_side))

    #plt.matshow(y, cmap='gray', vmin=0, vmax=10)
    #plt.show()

    #sys.exit()
    A = A.numpy()
    y = y.reshape((image_size * image_size, ))
    result = lsqr(A, y)
    x = result[0]  # Solution vector
    print("Solution x:", x)
    print("Residual norm ||Ax - y||:", result[3])  # result[3] is the norm of residual

    y_pred = A @ x
    #plt.hist(y_pred, bins=20)
    #plt.show()
    y_pred = y_pred.reshape((image_size, image_size))
    plt.matshow(y_pred, cmap='gray')
    plt.savefig(f'Results/Einstein_{n_pegs}_{image_size}.pdf')
    plt.show()


if __name__ == '__main__':
    main()