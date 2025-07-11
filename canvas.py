from math import sqrt
import torch, math
import matplotlib.pyplot as plt
from itertools import combinations
from skimage.draw import line_aa
from dataclasses import dataclass
import matplotlib.patches as patches




@dataclass
class Peg:
    label: int      # peg label (number) - pegs are numbered in counter-clockwise order
    x: int          # x position of the peg
    y: int          # y position of the peg
    pixel_label: int    # nr. of the pixel corresponding to the peg

@dataclass
class Pixel:
    label: int      # pixel label (number)
    x_center: float # x position of the center of the pixel
    y_center: float # y position of the center of the pixel
    x_label: int    # label of x position (integer number)
    y_label: int    # label of y position (integer number)


class Canvas:
    """
    Class for the Canvas to solve StringArt problem
    """

    def __init__(self, image_size, n_pegs, y_true, s):

        self.image_size = image_size    # nr. of pixel for one side of the image
        self.m_pixels = self.image_size * self.image_size  # total nr. of pixels in the image
        self.n_pegs = n_pegs
        self.s = s # superpixels per pixel-side: the total number of superpixels per pixel is self.s * self.s

        # PIXELS OF THE OUTPUT IMAGE:
        # Assign x_center, y_center, x_label, y_label for each pixel
        self.center_and_label_pixels_positions = ((i + 0.5, j + 0.5, i, j) for i in range(self.image_size * self.s) for j in range(self.image_size* self.s))
        # Initialization: all pixels are white
        self.pixels = [Pixel(i, x, y, l_x, l_y) for i, (x, y, l_x, l_y) in enumerate(self.center_and_label_pixels_positions)]

        # ANCHOR PEGS:
        self.anchor_pegs_positions = self.generate_pegs_positions()
        self.anchor_pegs = [Peg(i, x, y, pixel_pos) for i, (x, y, pixel_pos) in enumerate(self.anchor_pegs_positions)]

        # POSSIBLE THREADS COMBINATIONS
        self.possible_threads_combinations = list(combinations(range(n_pegs), 2))
        self.l = len(self.possible_threads_combinations)    # nr. of all possible edges

        # DEVICE:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # x, A, y, D, Identity matrix INITIALISATION:
        # X = vector of all possible edges (binary)
        self.x = torch.zeros(size=(self.l, ), dtype=torch.float32).to(self.device)
        # true vector representing target image
        self.y_true = torch.from_numpy(y_true).to(self.device)
        # map to go from edge space to pixel space
        self.A = self.matrix_A().to(self.device)
        # Identity matrix: useful for computing the optimisation step
        self.I = torch.eye(self.l).to(self.device)
        # Let's build the reduction matrix here:
        self.D = self.matrix_D().to(self.device)


    def generate_pegs_positions(self, ):
        """
        Such function generates pegs over the canvas in a circular fashion.
        """
        # Center and radius of the circumference where pegs are placed:
        center = self.image_size*self.s / 2
        radius = 0.5 *  self.image_size*self.s

        x_new, y_new, pixel_pos = 0, 0, 0
        # Let's collect pins positions:
        pin_positions = []
        for i in range(self.n_pegs):
            angle = 2 * math.pi * i / self.n_pegs      # ranges from 0 to 2*pi
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            best_distance_x = sqrt(2)
            best_distance_y = sqrt(2)
            for pix in self.pixels:
                # distance smaller than square root of 2 or previous assigned pixel
                if abs(pix.x_center - x) < best_distance_x and abs(pix.y_center - y) < best_distance_y:
                    x_new = pix.x_label
                    y_new = pix.y_label
                    pixel_pos = pix.label
                    best_distance_x = abs(pix.x_center - x)
                    best_distance_y = abs(pix.y_center - y)
            pin_positions.append((x_new, y_new, pixel_pos))
        return pin_positions


    def matrix_H(self):
        H = torch.zeros(size=(self.image_size, self.image_size), dtype=torch.float32)#.to(self.device)
        center = self.image_size / 2
        radius = 0.5 *  self.image_size
        for i in range(self.m_pixels):
            if sqrt((self.pixels[i].x_center - center)**2 + (self.pixels[i].y_center - center)**2) < radius:
                H[self.pixels[i].x_label, self.pixels[i].y_label] = 1.

        return H




    def matrix_D(self):
        """
        Builds a reduction matrix R of shape (m, m*s*s) to average over s*s rows.
        """
        rows = self.m_pixels * self.s * self.s
        D = torch.zeros((self.m_pixels, rows))

        for i in range(self.m_pixels):
            start = i * self.s * self.s
            end = start + self.s * self.s
            D[i, start:end] = 1 / (self.s * self.s)
        return D


    def matrix_A(self):
        """
        The matrix A is the core of image construction.
        Its task is to go from the space of all possible combinations of pegs to the canvas in which the number
        of pixels is increased by a factor s.
        Matrix A in high-resolution if self.s > 1.
        """
        A = torch.zeros(size=(self.m_pixels * self.s * self.s, self.l), dtype=torch.float32)
        for j, (i1, i2) in enumerate(self.possible_threads_combinations):
            p1_x, p1_y = self.anchor_pegs[i1].x, self.anchor_pegs[i1].y
            p2_x, p2_y = self.anchor_pegs[i2].x, self.anchor_pegs[i2].y
            # Anti-aliased line: rr, cc are arrays of pixel coordinates; val is an array of grayscale values
            rr, cc, val = line_aa(p1_x, p1_y, p2_x, p2_y)
            # Mask out any pixels outside the image boundaries
            mask = (rr >= 0) & (rr < self.image_size * self.s) & (cc >= 0) & (cc < self.image_size * self.s)
            rr, cc, val = rr[mask], cc[mask], val[mask]
            # Compute flattened 1D pixel indices
            val = torch.from_numpy(val)
            indices = rr * self.image_size * self.s + cc  # both rr and cc are 1D arrays
            # Assign the grayscale values into matrix A for this edge (column j)
            A[indices, j] += val  # val is also a 1D array, same length as indices

        return A


    def visualize_A(self) -> None :
        """

        """
        fig, ax = plt.subplots()
        for row in range(self.m_pixels):
            for col in range(self.l):
                color = 'black' if self.A[row][col] == 1 else 'white'
                # Normalize coordinates to [0,1]
                x = col / self.l
                y = row / self.m_pixels
                cell_width = 1 / self.l
                cell_height = 1 / self.m_pixels
                square = patches.Rectangle((x, y), cell_width, cell_height, facecolor=color, edgecolor='gray', linewidth=0.5)
                ax.add_patch(square)
        # Draw grid lines
        for x in torch.linspace(0, 1, self.l + 1):
            ax.axvline(x, color='black', linestyle='-', linewidth=0.5)
        for y in torch.linspace(0, 1, self.m_pixels + 1):
            ax.axhline(y, color='black', linestyle='-', linewidth=0.5)
        plt.show()


    def greedy_optimisation(self):
        """
        while true do:
            j = argmin_i ‖WF(x ± e_i) − Wy‖2
            ˜f = ∥WF( x ± e j ) − Wy ∥2
            if ˜f < f then
                x = x ± e j
                f = ˜f
            else
                break
            end
        end
        """

        def compute_loss(i: int, plus: bool):
            e_i = self.I[i]
            difference_vector = (self.D @ (self.A @ (self.x + e_i)).clamp_max(1.0) - self.y_true) if plus else (self.D @ (self.A @ (self.x - e_i)).clamp_max(1.0) - self.y_true)
            return torch.linalg.norm(difference_vector).item()

        old_error = 10000000.
        set_possible_indexes = set(range(self.l))

        # add edges
        while True:
            print("ciao sono in plus true")
            vector_losses = torch.tensor([compute_loss(i, plus=True) for i in set_possible_indexes])
            j = int(torch.argmin(vector_losses))
            print(j)
            set_possible_indexes -= {j}
            error = compute_loss(j, plus=True)
            if error < old_error:
                self.x[j] = 1
                old_error = error
            else:
                break

        # remove edges
        while True:
            print("ciao sono in plus false")
            vector_losses = torch.tensor([compute_loss(i, plus=False) for i in set_possible_indexes])
            j = int(torch.argmin(vector_losses))
            print(j)
            set_possible_indexes -= {j}
            error = compute_loss(j, plus=False)
            if error < old_error:
                self.x[j] = 1
                old_error = error
            else:
                break

        output = self.D @ (self.A @ self.x).clamp_max(1.0)

        return output
