import pandas as pd
import numpy as np


class PensionPremium(object):
    """
    Class to compute the premium for a given family
    with an invalid social worker. The family input is a dictionary
    of the following form {"name1": ["invalid",age, "M", True],
                           "name2": ["wife"/ "husband", age, "F", False],
                           "name3": ["son", age, "F"/"M", False],
                           ...,
                           "namen": [family_status, age, sex, Invalid? (T/F)]}
    In the last example "namen" gives the general form
    """

    def __init__(self, family, i_rate):
        self.database = "./pension_premium.xlsm"
        self.family = family
        self.i_rate = i_rate
        self.Px_table = pd.read_excel(self.database,
                                      sheetname="Px", index_col=0)
        self.PMGs = pd.read_excel(self.database)

    def convolution(self, X, Y):
        """
        Let X & Y be two distributions such that Omega(X) = Omega(Y).
        This function computes the convolution of X agains Y
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
    family = {"X": ["invalid", 50, "M", True],
              "Y": ["wife", 45, "F", False],
              "x1": ["son", 20, "M", False],
              "x2": ["son", 10, "F", False]}

    test = PensionPremium(family, None)
