# StringArt

## Name
String Art: Towards Computational Fabrication of String Images

## Description
This project was developed for the PhD exam course "Python Fundamentals" at the University of Milan. The code is based on the paper with the same title by Birsak et al. (DOI: [10.1111/cgf.13359](https://doi.org/10.1111/cgf.13359)).

The repository contains a Python implementation of a computational string art pipeline that reconstructs a target image by selecting thread connections between pegs arranged on a circular canvas.

## Visuals
StringArt problem and steps:

<img width="380" height="260" alt="Screenshot from 2025-07-21 11-16-07" src="https://github.com/user-attachments/assets/75d89910-58da-447d-ae71-ae3680ce7927" />

Original image:

<img width="300" height="300" alt="Einstein_quadrato" src="https://github.com/user-attachments/assets/5c576871-30d2-490c-b89c-4b82c19622de" />

Our result:

<img width="300" height="300" alt="1772091496783-7f295394-c805-474f-ac89-ddeecaefd7f5_1" src="https://github.com/user-attachments/assets/a56670da-88c1-4da3-a6a4-90db5f90389c" />

## Project structure
- `main.py`: entry point for loading a target image, building the canvas, and running the optimization.
- `canvas.py`: core implementation of the canvas, peg placement, matrices, and greedy optimization procedure.
- `utils.py`: image loading and preprocessing helpers.
- `Albert_Einstein.jpg`, `Taylor_Swift.jpg`: example input images included in the repository.

## Requirements
The project uses the following Python libraries:

- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-image`
- `scipy`
- `torch`

## Usage
Run the project from the repository root with:

```bash
python main.py
```

Before running it, make sure the target image path in `main.py` points to an image available in the repository or on your machine.

## Notes
- The current implementation places pegs on a circle and searches for thread combinations with a greedy optimization strategy.
- The output image is displayed with `matplotlib`, and the script also attempts to save a result PDF in a `Results/` directory.

## Project status
The project has been completed for the final exam of the course (approved in July 2025). The contributors have some ideas for future developments.
