from __future__ import division, print_function
import re
import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt
from sympy import diff, N, symbols, im, log


class polynom():
    # TODO: trigonometric functions & logarithms
    """
    Polynom class to sum, multiply, derive & integrate polynomials.
    """
    def __init__(self, expression):
        self.terms = self.exponents_dict(expression)
        self.expression = self.put_together(self.terms)

    def derive(self):
        """
        Method to derive the function given the expression
        :return: a polynom class with the derived polynomial
        """
        derivate_dict = {}

        for exponent in self.terms:
            if exponent != 0:
                derivate_dict[exponent-1] = self.terms[exponent]*(exponent)

        return polynom(self.put_together(derivate_dict))

    def integrate(self):
        """
        Method to integrate the function given
        :return: A polynom class with the integrated polynomial
        """
        integrate_dict = {}
        for exponent in self.terms:
            # Looking for  x^-1's, which have for integral ln(x)
            if exponent is not -1:
                n_coef = self.terms[exponent]/(exponent+1)
                integrate_dict[exponent+1] = n_coef
            else:
                integrate_dict["ln(x)"] = self.terms[exponent]

        result = self.put_together(integrate_dict)
        polynom(result)
        return polynom(self.put_together(integrate_dict))

    def put_together(self, eq_terms):
        """
        Method to show the equation in a string format
        :param eq_terms: the dictionary to be converted
        :return: a string in a polynomial way
        """
        # Getting the exponents and coeficients to merge by arranging
        exponents = [key for key in eq_terms]
        exponents.sort(reverse=True)
        coeficients = [eq_terms[coef] for coef in exponents]

        # Making them strings
        coeficients = np.array(coeficients, dtype=str)
        exponents = np.array(exponents, dtype=str)
        x_array = np.repeat("x^", len(coeficients))
        result = np.core.defchararray.add(coeficients, x_array)

        result = np.core.defchararray.add(result, exponents)
        result = np.ndarray.tolist(result)

        equation = ""
        not_first = False
        for term in result:
            # from a.0x^0 to a.0
            new_t = re.sub(r'(x\^0$)',"",term)
            # from a.0x^1, ax^1
            new_t = new_t.replace(".0", "")
            # from ax^1 to ax
            new_t = re.sub(r'x\^1$', "x", new_t)
            # from 1x^a to x^a
            new_t = re.sub(r'1x\^', "x^", new_t)

            # Add plus symbols if it is not the first element and it is not a negative term
            if not_first is True and new_t[0] is not "-":
                equation += " + " + new_t
            else:
                if new_t[0] == "-":
                    new_t = "- " + new_t[1:]
                equation += " " + new_t
                not_first = True
        return equation


    def to_terms(self, funct):
        """
        function to pass the polynomic string and convert it into a list with its factors
        as strings
        """
        clean = funct.replace(" ","")
        if clean[0] != "+" and clean[0] != "-":
            clean = "+"+clean
        # find any number followed by an x^ and a number, possibly fractional
        #  with or without a "-" in it or a number followed by an x or a number
        # alone or an x alone

        final = []
        vals = re.findall(r'\d*\.\d*?\w\^-?\d*\.?\d+?|' # eg. 2x^3.5
                          r'\d*x\^-\d+|'     # eg. 2x^-3
                          r'\d*\w\^\d+|'     # eg. 2x^3
                          r'\d\.\d+\w\^\d+|'  # eg. 0.22x^3
                          r'\w*\^\d*|'       # eg. x^3
                          r'\d+[xyz]|'       # eg. 23x
                          r'[xyz]|'          # eg. x
                          r'\d+|'            # eg. 23
                          r'\w*\([xyz]\)'    # eg. sinx
                          , clean)
        signs = re.findall(r'[+-]', clean)

        for i in range(len(vals)):
            final.append(signs[i] + vals[i])

        return final

    def exponents_dict(self, expression):
        """
        Method to pass a polynomial equation represented as a string.
        :return:  A dictionary with keys as exponents and values as coeficients
        """
        terms =  self.to_terms(expression)
        exp_value_dict = {}
        for term in terms:
            # If the first term after the symbol is zero, skip it.
            if term[1:] != "0" or term[1:] != "0x":
                symbol = term[0]
                # Get the coeficient at the beggining of the string it might be negative
                coeficient = re.findall(r'^\d*\.\d*|'
                                        r'^\d*', term[1:])[0]
                # If there was not result, then it was either a 1 from an x or a coeficient
                if coeficient == "":
                    coeficient = "1"
                # find the exponent value, which, should be at the end of the string (the term)
                # and can be any number of length 1 to 6. There may be a negative symbol between the
                # caret symbol and the number
                power_list = re.findall(r'(\^-?\d*\.?\d+?$)', term)
                # If there was no result, either it was a coeficient alone or a
                # letter alone
                if len(power_list) == 0:
                    # if there is an 'x' with no exponent: its power is 1, its coeficient
                    #  is the only number in it
                    if(len(re.findall(r'x', term))) == 1:
                        power = 1
                    else:
                        # if not, then it is a coeficient; its power = 0, its coeficient is the term
                        power = 0
                else:
                    # from power_list, get the only element (^+number) and take out the caret
                    power = power_list[0].replace("^","")
                exp_value_dict[float(power)] = float(symbol+coeficient)

        return exp_value_dict

    def evaluate(self, val):
        """
        This method will evaluate the polynom at the specified point
        :param val: The number to evaluate the function at
        :return: the corresponding f(x) at x
        """
        funct = ""
        len_dict = len(self.terms)
        i = 0
        for exp in self.terms:
            coef = str(self.terms[exp])
            power = str(exp)
            if i is 0:
                funct += coef + "*a**" +  power
            elif i > 0 and i < len_dict-1:
                funct += " +" + coef + "*a**" +  power
            else:
                funct += " +" + coef + "*a**" +  power
            i+=1

        # "a" will be passed to eval
        a = val
        final = eval(funct)

        return final


    def __add__(self, other):
        if other == 0:
            return self

        sum_dict = {}
        # For the terms of 'self' in other
        for term in self.terms:
            if term in other.terms:
                sum_dict[term] = other.terms[term] +  self.terms[term]
            else:
                sum_dict[term] = self.terms[term]


        # For the terms of 'other' not in sum_dict
        for val in other.terms:
            if val not in sum_dict:
                sum_dict[val] = other.terms[val]

        # Returning a polynom class
        return polynom(self.put_together(sum_dict))

    def __mul__(self, other):
        def coef_exp_mult(coef, exp, terms):
            # multiplier of the terms and sum of exponents
            return_dict = {}
            """
            This function will multiplicate a given coeficient and a
            exponent with a dictionary of terms.
            """
            for term in terms:
                # Summing exponents and multiplying coeficients
                sum_exp = exp + term
                mult_coef = coef*terms[term]
                return_dict[sum_exp] = mult_coef

            return return_dict

        elements = list()
        for term in self.terms:
            exponent = term
            coeficient = self.terms[term]
            sum_factor = coef_exp_mult(coeficient, exponent, other.terms)
            elements.append(sum_factor)

        test_final = [polynom(self.put_together(element)) for element in elements]

        final = polynom("0")
        for element in test_final:
            final += element

        return final

    def newton_roots(self, xi_init, desired_error = 10e-6):
        """
        Finding the roots of the function by the Newton-Raphson method
        :param xi: the initial value from where to start looking the root
        :param error: the minimum desired error of the estimated root and the real root
        :return: The approximate (or exact) value of the root
        """

        f = self
        xi = xi_init
        fi = f.evaluate(xi)
        fprime = f.derive()
        fprimei = fprime.evaluate(xi)
        xip1 = xi - fi/fprimei # x_i+1
        error = abs(xi - xip1)


        while (error > desired_error):
            xi = xip1
            fi = f.evaluate(xi)
            fprimei = fprime.evaluate(xi)
            xip1 = xi -  fi/fprimei
            error = abs(xi -  xip1)

        return xip1


def newton(f , xi = 1.1, desired_error = 10e-6, desv = 0.9, to_plot = True):
    """
    This function will compute the Newton-Rhapson
    aproximation method. Early Version, use with careful
    The format of the Newton-Rahpson is:
    xi, f(xi), f'(xi), xi+1 = xi - f(xi)/f'(xi), error = abs(xi+1 - xi)
    """
    # TODO: check sympy works
    #t = symbols("t")
    #fp = diff(f(t))


    xi = xi
    fx = f(xi)
    fx_prime = derivative(f, xi, 10e-10)
    #fx_prime = fp.subs({t: xi})
    xip1 = xi - fx/fx_prime
    error = abs(xi -  xip1)
    print(xi, fx, fx_prime)

    while error > desired_error:
        xi = xip1
        fx = f(xi)
        fx_prime = derivative(f, xi, 10e-10)
        # Check!
        #fx_prime = fp.subs({t: xi})
        xip1 = xi - fx/fx_prime
        error = abs(xi - xip1)
        if  abs(im(xip1)) > 0:
            print("No meaningful answer")
            return None

    if to_plot == True:
        # Graphing the aproximate location
        xip1 = float(xip1)
        x = np.linspace(xip1*(1-desv),xip1*(1+desv),100)
        plt.plot(x, f(x))
        plt.axvline(xip1, color = "r")
        plt.axhline(0.01, color = "#000000")
    else:
        print("Plot not asked")
    return N(xip1)


def bisection(funct, a, b, error = 1e-6):
    """
    Compute the roots of a given function
    :param f: the given function
    :param a: initial lower bound
    :param b: initial upper bound
    :param error: the desired error, set to 1e-6 by default
    """

    ai = a
    bi = b
    pi = (a+b)/2 # midpoint between a and b

    # if a and pi both have the same symbol,
    # then a = pi
    while True:
        if funct(ai)/abs(funct(ai)) == funct(pi)/abs(funct(pi)):
            ai = pi
            ptest = (ai + bi)/2
            if abs(pi - ptest) <= error:
                return ptest
            else:
                pi = ptest
        else:
            bi = pi
            ptest = (ai + bi)/2
            if abs(pi - ptest) <= error:
                return ptest
            else:
                pi = ptest
    return pi

def gaussJ(matrix,b = None):
    """
    This function will take a n*n matrix and compute the gauss jordan for
    said matrix
    :param matrix: The matrix to be tested upon
    :param b: Vector of results
    """
    bound = matrix.shape[0]
    index = 0
    rows = append_ab(matrix, b)
    # For every row in rows:
    for step in range(bound):
    #while True:
        injections = [i for i in range(bound)] # the row elements
        index = injections.pop(index)
        current_c = np.empty(rows.shape)
        # get the (i,i) element: H
        H = rows[index,index]
        # get the index row and divide by H, save it as a row: P_row (pivot row)
        p_row = rows[index]/H
        # appending the pivot row
        current_c[index] = p_row
        # delete said row and store it in a new (N-1)xN matrix : CA
        CA = np.delete(rows, index, axis = 0)
        # for every row j in CA ->
        row_pos = 0
        for rj in CA:
            # CA_j = CA_j[i]*P_row
            CA_j = rj - rj[index]*p_row
            current_c[injections[row_pos]] = CA_j
            row_pos+=1
        index += 1# Get a new row
        rows = current_c
        print(rows,"\n")
    return rows[:,bound]

def append_ab(mat, b):
    final = np.array([])
    index = 0
    bound = mat.shape[0]
    for element in mat:
        element = np.append(element,b[index])
        final = np.append(final, element)
        index += 1
    final.shape = (bound, bound+1)
    return final

def lu_decom(mat, b):
    pass

if __name__ == "__main__":
    print("Example testing")
    print("------POLYNOM------")
    f1 = polynom("x^2 + 36x - 5x^-2")
    print(f1.expression)
    print(f1.derive().expression)
    print(f1.integrate().expression)
    print()

    print("-----Gauss Jordan---")
    mat1 = np.random.randn(25)
    mat1.resize((2,2))
    b1 = np.random.randint(1,1000,2)

    vals = gaussJ(mat1, b1)
    print(vals)