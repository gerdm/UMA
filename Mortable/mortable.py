import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, exp


class Atable:
    # TODO: add more ways to calculate the table functions (lx, px)
    # TODO: add D'moivre, weibull and gompertz calculation
    # TODO: add mux approximation to empirical
    # TODO: add plotting options to empirical and makeham
    # TODO: change configurations for "single_plot"
    """
    the Atable class is an actuarial mortality table class. A csv file with 2 rows: age and
    mx must be introduced in the class.

    This class will perform a first analysis of the table and will compute any of the lx, dx, qx, & px.
    Note that a radix must be introduced to have proper lx values.
    """
    plot_names =    {"lx" : r"$l_x$",
                    "dx": r"$d_x$",
                    "qx": r"$q_x$",
                    "px": r"$p_x$",
                    "Lx": r"$L_x$",
                    "Tx": r"$T_x$",
                    "mx": r"$m_x$",
                    "e0x": r"$e^0_x$",
                    "ex": r"$e_x$"}

    columnsAsString = ", ".join(plot_names.keys())

    def __init__(self, table, reference = "mx"):
        self.table = pd.read_csv(table) # "table" must be a csv
        self.ref = reference
        self.meshed = False # False if values are not yet meshed
        self.approxs = {"makeham": [None,0]} # Stores the approximation function table and the maximum error

        # Setting age column
        self.table.rename(columns = {self.table.columns[0]:"age"}, inplace = True)
        ages = self.table.pop("age")
        self.table.index = ages




    def mesh(self, radix = 1000000):
        """
        Method to compute a mortality table, given the initial values
        :param radix: [optional] The theoretical values to calculate lx and dx
        :return:
        """
        funcs = ["lx","dx","px","qx","Lx","Tx","mx","e0x","ex"]


        if self.meshed == False:
            # If the initial value is a mx
            if self.ref == "mx":
                # Getting the values to add to the mesh

                #qx = 2mx/(2+mx)
                qx_vals = self.table[[self.ref]].apply(lambda x: 2*x/(2+x))
                #qx = 1-px
                px_vals = 1 - qx_vals
                lx_vals = np.array([radix])
                dx_vals = np.array([])

                # Computing the values for lx & dx
                i = 0
                for val in qx_vals.iterrows():
                    # As a discrete approximation: dx = lx*mx
                    dx_vals = np.append(dx_vals, round(float(val[1])*lx_vals[i]))
                    # -> l{x+1} = lx - d{x+1}
                    lx_vals = np.append(lx_vals, lx_vals[i]-dx_vals[i])
                    i += 1
                # Cutting the table one value
                lx_vals = np.delete(lx_vals, len(lx_vals)-1)

                # Lx = lx - (1/2)dx
                Lx_vals = lx_vals- (dx_vals/2)

                # Tx = sum from i = 0 to infty of L{x+i}
                Tx_vals = np.array([])
                for i in range(len(lx_vals)):
                    Tx_vals = np.append(Tx_vals, np.sum(Lx_vals[i:]))

                # e0x = Tx / lx
                e0x_vals = Tx_vals/lx_vals

                # ex = e0x - 1/2
                ex_vals = e0x_vals - 0.5

                self.table["qx"] = qx_vals
                self.table["px"] = px_vals
                self.table["lx"] = lx_vals
                self.table["dx"] = dx_vals
                self.table["Lx"] = Lx_vals
                self.table["Tx"] = Tx_vals
                self.table["e0x"] = e0x_vals
                self.table["ex"] = ex_vals

                self.table = self.table[funcs]
                self.meshed = True

    def muli_plot(self, output_name, plot_title = "Empirical Table", show = False):
        """
        Method to plot all functions at once
        :param output_name: The output name (specify name with ".[format]")
        :param plot_title: The main plot title
        :param show: Option to whether show or save the plot
        """

        # If the table has not been meshed, mesh!
        self.mesh()

        plt.style.use("bmh")

        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(self.table["lx"])
        ax[1,0].plot(self.table["dx"], "o")
        ax[1,0].plot(self.table["dx"], alpha = 0.5, color = "blue")
        ax[0,1].plot(self.table["px"])
        ax[0,1].plot(self.table["qx"])
        ax[1,1].plot(self.table["mx"])

        ax[0,0].set_title(r"$l_x$")
        ax[1,0].set_title(r"$d_x$")
        ax[0,1].set_title(r"$p_x  \ q_x$")
        ax[1,1].set_title(r"$m_x$")


        fig.suptitle(plot_title)

        if show:
            plt.show()
        else:
            fig.savefig(output_name)


    def to_csv(self, name):
        self.table.to_csv(name, index = False)


    def single_plot(self, column_name , output_name = "Empirical Table", show = False, approximation = False):
        """
        Method to calculate a single mortality table function.
        :param column_name: The mortality function to plot
        :param output_name: If necessary, the name of the file to be output
        :param show: Whether to show it or not. If True a file with the image will be saved.
        :param approximation: If given the name of an approximation, the value calculated will be from the approximation
        :return:
        """
        if column_name not in Atable.plot_names.keys():
            raise AttributeError

        # If the plot has not already been meshed, mesh!
        self.mesh()

        plt.style.use("bmh")
        fig, ax = plt.subplots()
        ax.plot(self.table[column_name]) # Plotting the appropriate column
        ax.set_title(Atable.plot_names[column_name]) # Plotting the appropriate plot name given the column name

        if show:
            plt.show()
        else:
            fig.savefig(output_name)


    def makeham(self, *funcs):
        """
        Calculate a theoretical mortatily table following Makeham
        :param funcs: a list of the functions to output in the mortality table
        """
        if self.approxs["makeham"][0] == None: # Table has not been calculated
            if len(funcs) == 0:
                funcs = ["lx","dx","px","qx","Lx","Tx","mx","e0x","ex","mux"]

            else:
                funcs = np.array(funcs)

            def Tx(lxv):
                tx_vector = np.array([])
                for i in range(len(lxv)):
                    tx_vector = np.append(tx_vector,sum(mCalc["Lx"](lxv)[0+i:]))
                return tx_vector

            mCalc = {"lx": np.vectorize(lambda k,s,g,c,x: round(k*(s**x)*g**(c**x))),
                     "dx": lambda lxv: np.abs(np.diff(lxv)),
                     "px": lambda lxv: lxv[1:]/lxv[:len(lxv)-1],
                     "qx": lambda lxv: 1- lxv[1:]/lxv[:len(lxv)-1],
                     "Lx": lambda lxv: (lxv[0:len(lxv)-1] + lxv[1:]),
                     "Tx": Tx,
                     "mx": lambda lxv: mCalc["dx"](lxv)/mCalc["Lx"](lxv),
                     "e0x": lambda lxv: mCalc["Tx"](lxv)/lxv,
                     "ex": lambda lxv: mCalc["e0x"](lxv) - 0.5,
                     "mux": np.vectorize(lambda k,s,g,c,x: -log(s)-log(g)*log(c)*c**x)
                    }

            str_keys = ", ".join(list(mCalc.keys()))
            for element in funcs:
                if element not in mCalc.keys():
                    raise KeyError("'{}' is not a valid function, the available functions are: {}".format(element, str_keys))

            # pandas dataframe with the commuted values
            commuted_ages = np.array([])
            base_age = 0

            # The value-vector with the theoretical C's
            sumC = 0
            # The value vector with the theoretical ln(g),
            #  where g = (Delta)2ln(lx)_1 /((C^x)(C_x^jumps -1 )^2)
            sumLng = 0
            # The value vector where with the theoretical ln(S),
            # where ln(S) = ((Delta)ln(lx)_1 - C^x*ln(g)*(C^jumps -1) / njumps
            sumLnS = 0

            sumLnk = 0
            # Numerator and denominator of final Cs
            sumNumeratorC = 0
            sumDenominatorC = 0

            self.mesh()
            # The number of parameters. Depends on the model chosen
            nparams = 4
            # The number of jumps for each bin and for each age.
            njumps = round(len(self.table.index) / nparams)

            # Computes the stratification of each group depending on the method. The resulting "strat_list" will contain
            # several lists which will contain each group stratification and each member of the group stratification
            # will be a list with age and the available lx at that age.
            strat_list = []
            age_counter = self.table.index[0]
            # for each of the strat groups
            #TODO: delete the appending of ages to the list: not useful
            for group in range(njumps):
                group_list = []
                strat_counter = 0
                # for each memeber of the group
                for memeber in range(nparams):
                    current_age = age_counter + njumps*strat_counter
                    group_list.append([current_age, self.table.ix[current_age,"lx"]])
                    strat_counter+=1
                age_counter+=1
                strat_list.append(group_list)

            for group in strat_list:
                loglx = np.array([]) # The list for the log(lx's)
                for element in group:
                    loglx = np.append(loglx, log(element[1]))
                loglxD1 = np.diff(loglx) # first difference of loglx
                loglxD2 = np.diff(loglxD1) # Second difference of loglx

                output_age = group[0][0]
                output_lx = group[0][1]
                C_val = (loglxD2[1]/loglxD2[0])**(1/njumps)
                lng_val = loglxD2[0] /( (C_val**output_age)*(C_val**njumps -1)**2 )
                lnS_val = (loglxD1[0] - C_val**output_age*lng_val*(C_val**njumps - 1))/njumps
                lnk_val = np.log(output_lx/(exp(lng_val)**(C_val**output_age)*exp(lnS_val)**output_age))

                commuted_ages = np.append(commuted_ages, output_age)

                base_age = commuted_ages[0]
                sumC += C_val
                sumLng += lng_val
                sumLnS += lnS_val
                sumLnk += lnk_val
                sumNumeratorC += lng_val*C_val**(base_age+1)
                sumDenominatorC += lng_val*C_val**(base_age)


            # f[val]: final val
            fk = exp(sumLnk/njumps)
            fs = exp(sumLnS/njumps)
            fc = sumNumeratorC/sumDenominatorC
            fg = exp((sumDenominatorC/njumps)/(fc**base_age))

            lxT = mCalc["lx"](fk,fs,fg,fc,self.table.index.values)

            # list comprehension to add the respective elements of the list inside a dict with keys as
            # the mortatility table function and value the list of element to add
            data_col = {element:lxT if element == "lx" else
                        mCalc[element](fk,fs,fg,fc,self.table.index.values) if element == "mux" else
                        mCalc[element](lxT) for element in funcs}


            # list comprehension to add the different elements in data_col, whose elements are not of equal
            # length
            makeham_table = pd.DataFrame(dict([(header,pd.Series(values)) for header,values in data_col.items()]))
            makeham_table.index.names = ["Age"]
            makeham_table.index = self.table.index
            #Ordering the table
            makeham_table = makeham_table[funcs]


            max_error = max(makeham_table["lx"] / self.table["lx"]- 1)
            print("Maximum 'lx' error: {}".format(round(max_error, 3)))

            ########## storing the table ##########
            self.approxs["makeham"][0] = makeham_table
            self.approxs["makeham"][1] = max_error
        else:
            print("Table already calculated, max error of: {}".format(self.approxs["makeham"][1]))



def main():
    # TODO: Complete selections, options and error handling
    import os
    # mayor.minor(a new def).mayor bug or correction.minor bug
    print("------MORTABLE v0.1.1.1------\n")

    files = os.listdir()
    files = [file for file in files if file.endswith(".csv")]
    ## If no files are found inside the working directory
    if len(files) == 0:
        print("""No .csv files were found in {}, please,\
change the path or insert .csv files in the folder""".format(os.path))
        return False

    print("Choose a file number to work with or type 'q' to quit")
    # Creates a selection of the available files
    indexes = range(len(files))
    selection = dict(zip(indexes, files))
    for sel in selection:
        print("{} ({}) ".format(sel, selection[sel]))


    ## FIRST FRAME ##
    sel = input("Selection: ")

    while sel not in [str(key) for key in selection]:
        if sel == "q":
            print("Quitting...")
            return None
        print("{} is not a valid option".format(sel))
        sel = input("Selection: ")

    sel = int(sel)
    file = selection[sel]
    print("{} selected".format(file))

    def plotting_menu():
        funSelect = input("Funtion to plot: ")
        while funSelect not in Atable.plot_names.keys():
            print("{} is not a valid function. The possible functions are: {}".format(funSelect, Atable.columnsAsString))
            funSelect = input("Funtion to plot: ")
        return funSelect

    def choosing_stage():
        ## SECOND FRAME ##
        wFile = Atable(file)
        opts = ["1","2","3","4","m","q"]
        functions =  """
            What would you like to do with {}?:
                1: Calculate empirical table and save as .csv
                2: Plot empirical lx, dx, px, qx & mx
                3: Plot a single empirical function
                4: Approximate empirical table to a model

                q: quit
                m: main menu
        """.format(file)
        print(functions)
        op = input("Selection: ")
        while op not in [key for key in opts]:
            print("{} is not an option".format(op))
            op = input("Selection: ")

        if op == "1":
            fileName = input("Output file name: ")
            wFile.to_csv(fileName)
            choosing_stage()


        elif op == "2":
            wFile.muli_plot("",show=True)
            choosing_stage()


        elif op == "3":
            fun_select = plotting_menu()
            wFile.single_plot(fun_select, show=True)
            choosing_stage()


        elif op == "4":
            # TODO: once added more approximations, choose which one.
            opts = ["1","2"]
            print("""
                Press:
                    1: Approximate table and save as .csv
                    2: plot a table function
            """)
            selection = ""
            while selection not in opts:
                selection = input("Selection: ")
                if selection == "m":
                    main()

            if selection == "1":
                out_name = input("Output Name (include '.csv' at the end): ")
                wFile.makeham()
                wFile.approxs["makeham"][0].to_csv(out_name)
                choosing_stage()
            if selection == "2":
                wFile.makeham()
                fun_select = plotting_menu()
                # TODO: appropiate selection of makeham plot
                choosing_stage()


        elif op == "m":
            main()
        else: # The 'q' was typed
            print("Quitting...")
            return None

    choosing_stage()

if __name__ == "__main__":
    main()