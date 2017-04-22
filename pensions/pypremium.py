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
                conv_dist = self.convolution(conv_dist, dist + extra_zeros)
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
        children_data = self.find_key_of("descendant")
        children_data = [self.family[membr] for membr in children_data]
        for child_data in children_data:
            prob_child = self.member_prob_x_to_k(child_data, k_years)
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
        # The probability of having 0, 1, ..., n children alive
        pension_sum = 0
        prob_children = self.convolute_children(years)
        for nsons, pr in enumerate(prob_children):
            pension_ix = self.pension_amount(nsons, spouse_alive)
            pension_sum += pr * pension_ix

        return pension_sum

    def find_key_of(self, key):
        # TODO: Use this function, replace get_children_data
        #      and recode convolute children
        """
        Return the keys of elements that have one given property
        of the family. key in ["invalid", "spouse", "descendant"]
        """
        elements = []
        for name, data in self.family.items():
            if data[0] == key:
                elements.append(name)

        return elements

    def member_prob_x_to_k(self, member_data, limit):
        member, age, sex, invalid = member_data

        kPx = 1
        # If a descendant is not invalid, then the pension stops paying once
        # he has gotten to age 24, since it can receive, at most, 25 payments
        # and the annuity is payed upfront
        if (member is "descendant") and (invalid is False) and (age + limit >= 25):
            kPx = 0
        # If it is not a descendant and invalid, the mortality table ends at 100, thus he can
        # only get payed up to 100 times, which accounts to 99 payments
        elif (member is not "descendant") and (invalid is True) and (age + limit >=100):
            kPx = 0
        # If it is not a descendant and NOT invalid, the mortality table ends at 110, thus he can
        # only get payed up to 109 times, which accounts to 109 payments
        elif (member is not "descendant") and (invalid is False) and (age + limit >=110):
            kPx = 0
    
        kPx *= self.prob_x_to_k(age, limit, sex, invalid)

        return kPx

    def pension_sum_element(self, period):
        """
        Compute a term of the sum that goes from k=0 to omega - x_j that
        takes account of the probability of the invalid being alive, the spouse
        (alive or dead) and the corresponding pensions for each of the children
        """
        invalid_data = self.family[self.find_key_of("invalid")[0]]
        spouse_data = self.family[self.find_key_of("spouse")[0]]

        invalid_prob = self.member_prob_x_to_k(invalid_data, period)
        spouse_prob = self.member_prob_x_to_k(spouse_data, period)
        pension_wife_alive = self.annuity_son_pension(period, True)
        pension_wife_dead = self.annuity_son_pension(period, False)
        Vk = self.V ** period

        term = invalid_prob * (spouse_prob * pension_wife_alive + \
                (1 - spouse_prob) * pension_wife_dead) * Vk

        return term

    def compute_premium(self):
        """
        Compute the premium to pay in order to insure the
        pensioner the rest of his life
        """
        omega = max(self.Px_table.index)
        youngest = min([self.family[x][1] for x in self.family])

        # The anticipated, monthly, and certain annuity
        annuity = (1 - self.V) / (1 + (1 + self.i_rate) ** (-1 / 12))

        premium = 0
        for k in range(omega - youngest):
            premium += self.pension_sum_element(k)

        premium *= annuity
        return premium

if __name__ == "__main__":
    family = {"X": ["invalid", 50, "M", True],
              "Y": ["spouse", 45, "F", False],
              "x1": ["descendant", 20, "M", False],
              "x2": ["descendant", 10, "F", False]}

    test = PensionPremium(family, 3000, 0.035, 2016)
    premium = test.compute_premium()
    print(premium)

