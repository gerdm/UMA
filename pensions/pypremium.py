import pandas as pd
import numpy as np


class PensionPremium(object):
    """
    Class to compute the premium for a given family
    with an invalid social worker. The family input is a dictionary
    of the following form {"invalid": [age, "M", True],
                           "wife": [age, "F", True],
                           "son1": [age, "F"/"M", True],
                           ...,
                           "sonn": [age, "F"/"M", True],}
    """

    def __init__(self, family, i_rate):
        self.family = family
        self.i_rate = i_rate

    def convolution(self, X, Y):
        """
        Let X & Y be two distributions such that Omega(X) = Omega(Y).
        This function computes the convolution between X & Y
        """
        x_dim = len(X)
        Amat = np.zeros([2 * x_dim - 1, x_dim])

        i = 0
        for x in X:
            for irange in range(x_dim):
                Amat[irange + i, irange] = x
            i += 1

        return Amat.dot(Y)


if __name__ == "__main__":
    test = PensionPremium(None, None)
    out = test.convolution(np.array([0.5, 0.1, 0.2, 0.2, 0]),
                           np.array([0.6, 0.2, 0.1, 0, 0.1]))
    print(out)
    