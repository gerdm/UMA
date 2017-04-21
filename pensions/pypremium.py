import pandas as pd
import numpy as np


class PensionPremium(object):
    """
    Class to compute the premium for a given family
    with an invalid social worker. The family input is a dictionary
    of the following form {"name1": ["invalid",age, "F"/"M", True],
                           "name2": ["spouse", age, "F"/"M", False],
                           "name3": ["descendant", age, "F"/"M", False],
                           ...,
                           "namen": [family_status, age, sex, Invalid? (T/F)]}
    In the last example "namen" gives the general form
    """

    def __init__(self, family, CB, i_rate, year):
        self.CB = CB
        self.year = year
        self.database = "./pension_premium.xlsm"
        self.family = family
        self.i_rate = i_rate
        self.Px_table = pd.read_excel(self.database,
                                      sheetname="Px", index_col=0)
        self.pmgs = pd.read_excel(self.database,
                                  sheetname="PMG", index_col=0)
        self.V = 1 / (1 + i_rate)

    def convolution(self, X, Y):
        """
        Let X & Y be two distributions such that Omega(X) = Omega(Y).
        This function computes the convolution of X agains Y
        """
        x_dim = len(X)
        Amat = np.zeros([2 * x_dim - 1, x_dim])
        Bmat = np.array(Y)

        i = 0
        for x in X:
            for irange in range(x_dim):
                Amat[irange + i, irange] = x
            i += 1

        return Amat.dot(Bmat)

    def pension_amount(self, number_sons, spouse_alive):
        """
        Compute the pension to pay according to the number
        of sons and if the pensioner has or has not an alive spouse
        """
        pmg = self.pmgs.ix[self.year, "PMG"]

        if number_sons == 0 and spouse_alive is False:
            adjustment_factor = 0.15
        else:
            adjustment_factor = number_sons * 0.10 + spouse_alive * 0.15

        return max(self.CB * (1 + adjustment_factor), pmg)

    def map_morttable(self, sex, invalid):
        """
        Return the name of the column to map to
        the mortality table.
        :param sex: "M" or "F"
        :param invalid: True (if invalid) or False
        """
        sex_dict = {"M": "male", "F": "female"}

        if invalid:
            status = sex_dict[sex] + "_invalid"
        else:
            status = sex_dict[sex] + "_active"

        return status

    def prob_x_to_k(self, x, k, sex, invalid):
        """
        Compute the probability that a person of age x
        reaches x + k alive. This is given by the multiplication
        of k-1 factors: Px * Px+1 *  ... * Px+k-1
        """
        col_map = self.map_morttable(sex, invalid)

        if k == 0:
            return 1
        else:
            probs = self.Px_table.ix[x: x + (k - 1), col_map]
            return np.prod(probs)

    def get_children_data(self):
        """
        Return a list with tuples of (age, sex, status)
        for each children in the family
        """
        children_data = []
        for member in self.family:
            member_elements = self.family[member]
            if member_elements[0] is "descendant":
                children_data.append(tuple(member_elements[1:]))

        return children_data

    def bernoulli_convolutions(self, distributions):
        """
        Given a set of bernoulli distributions P({0, 1}) = 1,
        convolute them two by two. If there is only one element,
        return the distribution
        """
        conv_dist = None
        number_elements = len(distributions)
        if number_elements > 1:
            # Convolute the first two elements
            conv_dist = self.convolution(distributions[0],
                                         distributions[1])
            # For-loop will work only if number_elements
            # is greater than 3
            for dist in distributions[2:]:
                # Add dummy zeroes to complete the convolution
                # computation
                number_zeros = len(conv_dist) - 2
                extra_zeros = [0 for i in range(number_zeros)]
                conv_dist = self.convolution(conv_dist,
                        dist + extra_zeros)
                # Remove the added dummy zeros
                conv_dist = conv_dist[:-number_zeros]
                
        else:
            conv_dist = distributions[0]

        return conv_dist

    def convolute_children(self, k_years):
        """
        Return the probability distribution of the convolution
        of livelyhood of children up to (Xi + k) years; where
        Xi is the current age of the ith child
        :param k_years: number of ages to live
        """
        prob_distribution = []
        children_data = self.get_children_data()
        for child_data in children_data:
            age, sex, iv_status = child_data
            prob_child = self.prob_x_to_k(age, k_years, sex, iv_status)
            # For each child exists either a probability of
            # being either alive or dead
            prob_distribution.append([1 - prob_child, prob_child])

        prob_convol = self.bernoulli_convolutions(prob_distribution)

        return prob_convol

    def annuity_son_pension(self, years, spouse_alive):
        """
        Compute the factor of the pension premium that accounts
        for the probability of live children and whether or not
        the spouse is alive.
        :param years: number of years of expectancy in livelyhood
        :param spouse_alive: whether or not the spouse is alive
        """
        pass


if __name__ == "__main__":
    family = {"X": ["invalid", 50, "M", True],
              "Y": ["spouse", 45, "F", False],
              "x1": ["descendant", 20, "M", False],
              "x2": ["descendant", 10, "F", False]}

    test = PensionPremium(family, 3000, 0.35, 2016)
    print(test.convolute_children(1))
