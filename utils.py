
import logging
from typing import Tuple

import numpy as np
from scipy import optimize
from skimage.color import gray2rgb
# import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist


# Start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


def apply_color_lut(image, color: str, do_equalize_hist=True):
    multiplier = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    if do_equalize_hist:
        image = (equalize_hist(image) * 200).astype(np.uint8)
    return gray2rgb(image) * multiplier[color]


def to_color(image, color: str, do_equalize_hist=True):
    multiplier = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    if do_equalize_hist:
        image = (equalize_hist(image) * 200).astype(np.uint8)
    return gray2rgb(image) * multiplier[color]


def fit_circle(X: np.ndarray,
               Y: np.ndarray,
               center_estimate: Tuple[np.number, np.number]):
    return CircleFitter(X, Y, center_estimate).fit()


class CircleFitter:
    """Fit a circle using scipy least squares method

    https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
    """

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 center_estimate: Tuple[np.number, np.number]):
        self.X = X
        self.Y = Y
        self.center_estimate = center_estimate

    def calc_R(self, xc: float, yc: float):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((self.X - xc)**2 + (self.Y - yc)**2)

    def f_2(self, c: Tuple[float, float]):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    def Df_2(self, c: Tuple[float, float]):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq
        """
        xc, yc = c
        df2b_dc = np.empty((len(c), self.X.size))

        Ri = self.calc_R(xc, yc)
        df2b_dc[0] = (xc - self.X)/Ri  # dR/dxc
        df2b_dc[1] = (yc - self.Y)/Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc.T

    def fit(self):
        center_estimate = self.center_estimate  # get centroid from shapely
        results = optimize.least_squares(self.f_2, center_estimate, jac=self.Df_2)
        center_2b = results.x

        Ri_2b = self.calc_R(*center_2b)
        R_2b = Ri_2b.mean()
        # residu_2b = sum((Ri_2b - R_2b)**2)
        return center_2b, R_2b


def main():
    X = np.array([9, 35, -13, 10, 23, 0])
    Y = np.array([34, 10, 6, -14, 27, -10])
    x_m, y_m = X.mean(), Y.mean()
    fitter = CircleFitter(X, Y, (x_m, y_m))

    print(fitter.fit())


if __name__ == "__main__":
    main()
