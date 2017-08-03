import numpy as np
from scipy import linalg
from itertools import permutations
from numpy.linalg import LinAlgError
from prettytable import PrettyTable

class BasicSol(object):
    def __init__(self, A, b, Z):
        self.A = A
        self.b = b
        self.Z = Z
        
    def basic_solutions(self):
        """Calculate the basic solution at each extreme point
        :param funct: the linear function to evaluate
        :param extreme_points: an array of list of points to evaluate"""
        solutions = []
        for basic_sol in self.extreme_points():
            solutions.append(self.Z(*basic_sol)) # * unpacks the list
    
        return solutions
    
    def extreme_points(self):
        """ Returns the position of the variables and the Z values for each possible basic solution
        :params xi: Number of variables
        :params A: Matrix NxM
        :Params b: solutions
        """
        extreme_possible_points = []
        ncols = self.A.shape[1] # Number of variables
        indices_list = self.permutation_list(ncols, len(self.b))

        for index in indices_list:
            solution = np.zeros(ncols)
            submatrix = self.A[:,index]

            try:
                subm_ans = linalg.solve(submatrix, self.b)
            except LinAlgError:
                print("Singular matrix at: {}".format(submatrix))

            ians = 0 # index of the answer
            for p in index:
                solution[p] = subm_ans[ians]
                ians += 1

            if sum(solution < 0) == 0:
                extreme_possible_points.append(solution)

        return extreme_possible_points
    
    def permutation_list(self, xi, len_b):
        """List of indices to be taken from the original Matrix
        :param xi: Number of variables
        :param len_b: Number of solutions
        """
        indices = []
        for element in permutations(range(xi), r = len_b):

            # for every two indices i,j; i<j in the list => element[i] < element[j]
            possible_perm = True
            last_value = -1
            for value in element:
                if last_value > value: # not a possible permutation
                    possible_perm = False
                    break
                else:
                    last_value = value

            if possible_perm == True:    
                indices.append(list(element))

        return indices
    
    def maximizer(self):
        max_z = max(self.basic_solutions())
        index_max = self.basic_solutions().index(max_z)
        basic_sol = self.extreme_points()[index_max]
        return basic_sol, max_z




class Simplex(object):
    """Matrix resolution of the simplex algorithms"""
    def __init__(self, A, b, coefs = 0):
        """
        :param A: Constraint matrix of dimensionality MxN
        :param b: Vector of solutions
        :param coef: vector of coeficients
        """
        self.A = A
        self.coefs = coefs
        self.b = b

    def BigM(self, M, c):
        """

        :param M: The BIG M
        :param c: The "M" multiplier at the left part of the new equation
        :return: Simplex solution
        """
        part_sol = self.Simplex()
        return c*M - part_sol

    def Simplex(self):
        """Solve the Simplex method"""
        all_non_negatives = lambda solution: True if np.all([el >= 0 for el in solution]) == True else False
        #is_optimal_sol = lambda c_hat, coefs, sol: True if len(np.where(c_hat[:len(self.b[0])-1] > 0)[0]) == 0 and all_non_negatives(sol) == True else False
        is_optimal_sol = lambda c_hat, coefs: True if len(np.where(c_hat[:len(self.b[0])-1] > 0)[0]) == 0 else False

        Bm1A = self.A; Bm1A = Bm1A.astype(float) # Matrix B^-1A

        Bm1b = self.b.T # Matrix B^-1b

        iteration_index = self.get_cjs(Bm1A) # the indices of the variables that come into the simplex
        iteration_index = iteration_index.astype(int)
        ctB = self.get_iteration_coef(self.coefs, iteration_index) # The coefficients per iteration for basic vars

        c_hat = self.coefs - np.dot(ctB, Bm1A)
        z = np.dot(ctB, Bm1b) # Validate the optimal

        self.print_simplex(Bm1A, ctB, Bm1b, c_hat, z)

        while not is_optimal_sol(c_hat, Bm1b):
            # The variable that gets into the simplex
            entering_variable = np.argmax(c_hat)
            #The variable out of the simplex
            leaving_variable = self.get_leaving_variable(Bm1A[:,entering_variable], Bm1b.T[0])

            Bm1A, Bm1b = self.make_base(Bm1A, Bm1b, leaving_variable, entering_variable)

            iteration_index = self.get_cjs(Bm1A) # the index of the variables that come into the simplex
            iteration_index = iteration_index.astype(int)
            ctB = self.get_iteration_coef(self.coefs, iteration_index) # The coefficients per iteration for basic vars

            c_hat = self.coefs - np.dot(ctB, Bm1A)
            z = np.dot(ctB, Bm1b) # Validate the optimal

            if all_non_negatives(Bm1b):
                self.print_simplex(Bm1A, ctB, Bm1b, c_hat, z)

        return z[0]

    def print_simplex(self, A, coefs, b, chat, z):
        round_n = 3
        x_vars = ["x"+str(round(x, round_n)) for x in range(A.shape[1])]
        x_vars.insert(0, "Ctb")
        x_vars.append("Cs")
        simplex_table = PrettyTable(x_vars)
        for i in range(self.A.shape[0]):
            slist = []
            slist.append(coefs[i])
            for element in A[i]:
                slist.append(round(element,round_n))
            slist.append(round(b[i].tolist()[0], round_n))

            simplex_table.add_row(slist)


        last_row = chat.tolist()[0]
        last_row = [round(el, round_n) for el in last_row]
        last_row.insert(0, "Cbar")
        last_row.append(round(z[0],round_n))

        simplex_table.add_row(last_row)
        xpos = self.get_cjs(A).tolist()
        xpos = ["x"+str(int(x)) for x in xpos]
        xpos.append("-Z-")
        simplex_table.add_column("Ctx",xpos)
        print(simplex_table)

    def get_cjs(self, A):
        """Given any MxN matrix, the function will find the indices in which
        an identity matrix is formed.
        :param A: MxN matrix
        :return: An array of indices where an MxM indentity matrix is formed; False if there exists no such matrix
        """
        A_nrows, A_ncols = A.shape
        index_array = np.zeros(A_nrows)
        current_index = 0

        for col in range(A_ncols):
            current_column = A[:, col]
            if np.sum(abs(current_column)) == 1: # There can only exist one element
                position = np.where(current_column == 1)
                index_array[position] = current_index
            current_index += 1

        return index_array if sum(abs(index_array)) > 0 else False


    def get_iteration_coef(self, coeficients, index_array):
        """Returns coeficients of the varibles in the simplex
        :param coefficients: the complete vector of coeficients
        :param index_array: the vector of the indices into the simplex
        :return: List with the values of the coeficients into the simplex
        """
        iteration_coefs = []
        for val in index_array:
            iteration_coefs.append(coeficients[0][val])

        return iteration_coefs

    def get_leaving_variable(self, entering_row, constants):
            """Return the index row that goes out of the Simplex"""
            with np.errstate(divide = "ignore", invalid = "ignore"): # ignoring x/0 and 0/0

                possibles = constants.T / entering_row
                minv = min(possibles[possibles > 0])
                return np.where(possibles == minv)[0][0]

    def make_base(self, matrix, b_const, row_index, col_index):
        """Given a row and a column index, make all the row equals to zero
        except the intersection @ (row_index, col_index)
        :param matrix: Matrix to change
        :param b_const: The vector of consants
        :param row_index: Row of the pivot
        :param col_index: Column of the pivot
        :return a tuple with the new matrix and the new vector of contants"""
        b_const.shape = (matrix.shape[0], 1) # Vector Nx1
        new_matrix = np.append(matrix, b_const, axis = 1)


        mat_rows, mat_cols = matrix.shape
        pivot = new_matrix[row_index, col_index]

        # Setting the nth_row, mth_col = 1
        new_row = new_matrix[row_index]/pivot

        new_matrix[row_index] = new_row

        for i in range(mat_rows):
            if i != row_index:
                target_ncol = col_index

                target_val = new_matrix[i, target_ncol]

                new_matrix[i] = new_matrix[i] - target_val*new_matrix[row_index]

        return new_matrix[:,:matrix.shape[1]], new_matrix[:,matrix.shape[1]:]


class Transportation(object):
    def __init__(self, cost_matrix, supply, demand):
        self.cost_matrix = cost_matrix
        self.demand = demand
        self.supply = supply
        self.start_solution = self.northwest()

    def loop(self, row_pos, col_pos):
        #TODO: finish method.
        pass

    def is_factible(self, it_matrix):
        """
        Solve the transportation problem.
        :param it_matrix: an iterable matrix of possible solutions
        :return: the factible solution
        """

        supply_ch = [np.nan for el in range(len(self.supply))] # change in supply
        demand_ch = [np.nan for el in range(len(self.demand))] # change in demand
        supply_ch[0] = 0

        # Filling Ci and Vi
        for i in range(len(supply_ch)):
            for j in range(len(demand_ch)):

                if not np.isnan(it_matrix[i,j]):
                    if np.isnan(supply_ch[i]) and not np.isnan(demand_ch[j]):
                        supply_ch[i] = self.cost_matrix[i, j] - demand_ch[j]
                    elif np.isnan(demand_ch[j]) and not np.isnan(supply_ch[i]):
                        demand_ch[j] = self.cost_matrix[i, j] - supply_ch[i]

        # Validating <=
        solutions = []
        coordinates = []
        for i in range(len(supply_ch)):
            for j in range(len(demand_ch)):
                if np.isnan(it_matrix[i,j]):
                    coordinates.append((i+1,j+1))
                    solution = supply_ch[i] + demand_ch[j] - self.cost_matrix[i,j]
                    solutions.append(solution)

        # TODO: remove use of final_coords, iterate automatically until arrive at a solution
        final_coords = [v for v in zip(solutions, coordinates)] # coordinates to evaluate for next iteration
        # Printing the coordinates with values for the next iteration
        for row in final_coords:
            print(row)

        for sol in solutions:
            if sol > 0:
                return False
        else:
            return True



    def northwest(self):
        """
        Method to dump the values in the transportation matrix
        :return: A matrix with a factible solution
        """
        resources = np.empty(self.cost_matrix.shape)
        resources[:] = np.nan

        supply_v = self.supply.copy()
        demand_v = self.demand.copy()

        supply_row = 0
        demand_col = 0


        while sum(supply_v) != 0 and sum(demand_v) != 0:
            current_demand = demand_v[demand_col]
            current_supply = supply_v[supply_row]
            amount_taken = min(current_supply, current_demand)# The amount  supply or demanded

            resources[supply_row, demand_col] = amount_taken

            demand_v[demand_col] -= amount_taken
            supply_v[supply_row] -= amount_taken

            #print(resources, "\n")

            if demand_v[demand_col] == 0 and supply_v[supply_row] == 0:
                # Jump either a row or a column
                row_col_selector = np.random.randint(0,1)
                supply_row += row_col_selector
                demand_col += 1 - row_col_selector

            elif supply_v[supply_row]== 0:
                supply_row += 1

            else:
                demand_col += 1

        return resources


def hungarian(hmat):
    """Solves the first 2 iterations of the hungarian algorithm
    :param hmat: the hungarian matrix to solve
    :return: None, it prints out the first two iterations.
    If the matrix contains non feasible values, add "none" from numpy
    """

    finalm = hmat.copy()
    n = finalm.shape[0]
    for i in range(n):
        finalm[i] = finalm[i] - np.nanmin(finalm[i])
    print(finalm, "\n")

    for i in range(n):
        finalm.T[i] = finalm.T[i] - np.nanmin(finalm.T[i])
    print(finalm)

if __name__ == "__main__":
    from numpy import nan
    h_matrix = np.array([[22,  18,  30,  18, 0],
                         [18,  nan, 27,  22, 0],
                         [26,  20,  28,  28, 0],
                         [16,  22,  nan, 14, 0],
                         [21,  nan, 25,  28, 0]])
    hungarian(h_matrix)