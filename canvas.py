import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from skimage.draw import line
from dataclasses import dataclass
import matplotlib.patches as patches




@dataclass
class Peg:
    label: int      # peg label (number) - pegs are numbered in counter-clockwise order
    color: str      # peg label (color)
    x: int          # x position of the peg
    y: int          # y position of the peg
    pixel_label: int    # nr. of the pixel corresponding to the peg

@dataclass
class Pixel:
    label: int      # pixel label (number) 
    color: str      # pixel color
    x_center: float # x position of the center of the pixel
    y_center: float # y position of the center of the pixel
    x_label: int    # label of x position (integer number)
    y_label: int    # label of y position (integer number)


class Canvas:
    """
    class for the Canvas to solve StringArt problem
    """

    def __init__(self, image_size, n_pegs):

        self.image_size = image_size    # nr. of pixel for one side of the image
        self.n_pixels =  self.image_size * self.image_size  # total nr. of pixels in the image
        self.n_pegs = n_pegs

        # PIXELS OF THE OUTPUT IMAGE:
        # assign x_center, y_center, x_label, y_label for each pixel
        self.center_and_label_pixels_positions = ((i + 0.5, j + 0.5, i, j) for i in range(self.image_size) for j in range(self.image_size))
        # initialization: all pixels are white
        self.pixels = [Pixel(i, 'white', x, y, l_x, l_y) for i, (x, y, l_x, l_y) in enumerate(self.center_and_label_pixels_positions)]

        # ANCHOR PEGS:
        self.anchor_pegs_positions = self.generate_pegs_positions()
        self.anchor_pegs = [Peg(i, 'green', x, y, pixel_pos) for i, (x, y, pixel_pos) in enumerate(self.anchor_pegs_positions)]

        # POSSIBLE THREADS COMBINATIONS
        self.possible_threads_combinations = list(combinations(range(n_pegs), 2))
        self.l = len(self.possible_threads_combinations)    # nr. of all possible edges

        # x, A, y initialisation:
        # X = vector of all possible edges (binary)
        self.x = torch.zeros(size=(self.l, ), dtype=torch.float64,  requires_grad=True)
        # y_predicted = vector of pixels of the approximated image (real values in [0,1])
        self.y_predicted = torch.zeros(size=(self.n_pixels, ), dtype=torch.float64,  requires_grad=True)
        # map to go from edge space to pixel space
        self.A = torch.zeros(size=(self.n_pixels, self.l), dtype=torch.float64,  requires_grad=False)


    def generate_pegs_positions(self, ):
        # Center and radius of the circumference where pigs are placed:
        center = self.image_size / 2
        radius = 0.5 *  self.image_size

        x_new, y_new, pixel_pos = 0, 0, 0
        # Let's collect pins positions:
        pin_positions = []
        for i in range(self.n_pegs):
            angle = 2 * np.pi * i / self.n_pegs      # ranges from 0 to 2*pi
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            best_distance_x = 0.707107
            best_distance_y = 0.707107
            for pix in self.pixels:
                # distance smaller than square root of 2 or previous assigned pixel
                if abs(pix.x_center - x) <= best_distance_x and abs(pix.y_center - y) <= best_distance_y:
                    x_new = pix.x_label
                    y_new = pix.y_label
                    pixel_pos = pix.label
                    best_distance_x = abs(pix.x_center - x)
                    best_distance_y = abs(pix.y_center - y)
            pin_positions.append((x_new, y_new, pixel_pos))
        return pin_positions


    def adjust_color(self, ith_pixel: int, color: str) -> None:
        pass

    def update_canvas(self):
        pass


    def matrix_A(self):
        for j, (i1, i2) in enumerate(self.possible_threads_combinations):
            # print('Pixels:', self.possible_threads_combinations[j])
            p1_x, p1_y, p2_x, p2_y = self.anchor_pegs[i1].x, self.anchor_pegs[i1].y, self.anchor_pegs[i2].x, self.anchor_pegs[i2].y
            rr, cc = line(p1_x, p1_y, p2_x, p2_y)
            # Clip to valid image bounds
            rr = np.clip(rr, 0, self.image_size - 1)
            cc = np.clip(cc, 0, self.image_size - 1)
            # flattening 2D indices to 1D and marking pixel coverage:
            indices = rr * self.image_size + cc
            # print(indices)
            self.A[indices, j] = 1.0
        return self.A


    def visualize_A(self):
        fig, ax = plt.subplots()
        for row in range(self.n_pixels):
            for col in range(self.l):
                color = 'black' if self.A[row][col] == 1 else 'white'
                # Normalize coordinates to [0,1]
                x = col / self.l
                y = row / self.n_pixels
                cell_width = 1 / self.l
                cell_height = 1 / self.n_pixels
                square = patches.Rectangle((x, y), cell_width, cell_height, facecolor=color, edgecolor='gray', linewidth=0.5)
                ax.add_patch(square)
        # Draw grid lines
        for x in np.linspace(0, 1, self.l + 1):
            ax.axvline(x, color='black', linestyle='-', linewidth=0.5)
        for y in np.linspace(0, 1, self.n_pixels + 1):
            ax.axhline(y, color='black', linestyle='-', linewidth=0.5)
        plt.show()
