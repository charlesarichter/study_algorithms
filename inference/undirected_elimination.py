""" Here's a docstring! """

import numpy as np
from sets import Set
from tablend import TableND

def getprodmembers_deprecated(factor1, factor2):
    """ Compute the variables involved in the product of factor1 and factor2
    along with their domains """

    # First figure out which variables we are going to output
    output_set = Set([])
    output_vars = []
    output_domains = []
    overlap_set = []

    for i in xrange(len(factor1.whichvars)):
        # print factor1.whichvars[i]
        
        var = factor1.whichvars[i]
        dom = factor1.domains[i]

        if var not in output_set:

            # Add to the set
            output_set.add(var)
            
            # Also add to an (ordered list)
            output_vars.append(var)

            # Also add the size of this variable's domain
            output_domains.append(dom)

    # for i in xrange(len(factor2.whichvars)):
    #     # print factor2.whichvars[i]
    #     output_set.add(factor2.whichvars[i])

def factorprod_deprecated(factor1, factor2):
    """ Takes in two factors and returns the factor product of the two.
    Output will be a factor joining a set of variables that is the union of
    the variables involved in the two input factors. """
    
    print "Factor Product"

    # First figure out which variables we are going to output
    output_set = Set([])

    for i in xrange(len(factor1.whichvars)):
        # print factor1.whichvars[i]
        output_set.add(factor1.whichvars[i])
    for i in xrange(len(factor2.whichvars)):
        # print factor2.whichvars[i]
        output_set.add(factor2.whichvars[i])

    # List of the variables that are represented in the output CPT
    out_vars = list(output_set)

    # Output variable domains
    print out_vars

    # Get the sizes of domains of the variables


    # Make an empty CPT
    # cpt = np.zeros(

    # Now loop over the output variables to compute products

# # Normally, here we would do something like:
#     for var1 in xrange(1):
#         for var2 in xrange(1):
#             for var3 in xrange(1):
#                # Compute product here
# But we can't do this, because we don't know which vars to loop over...

    # for x in np.nditer(factor2.cpt):
    #     print x 

    # print factor1.get_cpt_size()
    # print factor1.get_cpt_shape()

    for i in xrange(factor1.get_cpt_size()):

        varidents1 = factor1.get_var_idents(i)
        # print "Which variables for factor 1: " 
        # print factor1.whichvars
        # print "Values of factor 1 variables: "
        # print varidents1

        for j in xrange(factor2.get_cpt_size()):
            # print "i = ", i, ", j = ", j 

            varidents2 = factor2.get_var_idents(j)

            print ""
            print "Which variables for factor 1: " 
            print factor1.whichvars
            print "Values of factor 1 variables: "
            print varidents1
            print "Which variables for factor 2: " 
            print factor2.whichvars
            print "Values of factor 2 variables: "
            print varidents2
            raw_input("pause")

            
# def factortbd(f1_whichvars,f1_values,f2_whichvars,f2_values):
#     """ TBD """
#
#     print ""
#     print "Which variables for factor 1: " 
#     print f1_whichvars
#     print "Values of factor 1 variables: "
#     print f1_values
#     print "Which variables for factor 2: " 
#     print f2_whichvars
#     print "Values of factor 2 variables: "
#     print f2_values
#
#     # a = set(f1_whichvars) & set(f2_whichvars)
#     # print a
#
#     # a = Set([])
#     # for i in xrange(len(f1_whichvars)):
#     #     a.add(f1_whichvars[i])
#     # for j in xrange(len(f2_whichvars)):
#     #     a.add(f2_whichvars[j])
#     #
#     # print a
#     #
#
#     raw_input("pause")


def factorprod(factor1, factor2):
    """ Takes in two factors and returns the factor product of the two.
    Output will be a factor joining a set of variables that is the union of
    the variables involved in the two input factors. """
    
    """ Running list of Python-specific structures and functions used here:
        - list.index()
        - Set
        """

    """ Get a list of variables involved """
    output_vars = Set([])
    for i in xrange(len(factor1.whichvars)):
        output_vars.add(factor1.whichvars[i])
    for j in xrange(len(factor2.whichvars)):
        output_vars.add(factor2.whichvars[j])
    print output_vars

    """ Get a list of variables that overlap between the two factors """
    overlap_vars = list(set(factor1.whichvars) & set(factor2.whichvars))
    print overlap_vars

    """ For each overlap variable, record which dim of each cpt is involved """
    overlap_dims = []
    for i in xrange(len(overlap_vars)):
        d1 = factor1.whichvars.index(overlap_vars[i])
        d2 = factor2.whichvars.index(overlap_vars[i])
        # print d1
        # print d2
        overlap_dims.append((d1,d2))



    """ Iterate over all elements of both factor CPTs """
    for i in xrange(factor1.get_cpt_size()):
        variable_values1 = factor1.get_variable_values(i)
        for j in xrange(factor2.get_cpt_size()):
            variable_values2 = factor2.get_variable_values(j)
            
            """ if factor1 and factor2 agree on the value of the variable for
            this i and j, then multiply the cpt entries and put the result into
            the correct place in the output cpt.

            otherwise, just move on """


            """ See whether both factors agree on the value of overlap vars """
            agree = True
            for j in xrange(len(overlap_vars)):
                # print 'Looking at overlapping variable: ', overlap_vars[j]
                # print 'Dim in factor 1: ', overlap_dims[j][0]
                # print 'Dim in factor 2: ', overlap_dims[j][1]

                v1 = variable_values1[overlap_dims[j][0]]
                v2 = variable_values2[overlap_dims[j][1]]

                # print v1
                # print v2
                if v1 != v2:
                    agree = False

            if not agree:
                continue

            print ""
            print "Which variables for factor 1: " 
            print factor1.whichvars
            print "Values of factor 1 variables: "
            print variable_values1
            print "Which variables for factor 2: " 
            print factor2.whichvars 
            print "Values of factor 2 variables: "
            print variable_values2
            raw_input("pause")


class Node:
    """ Probably need a comment here """
    
    def __init__(self):
        self.value = 0;

class Factor(object):
    """ Factor class takes whichvars and cpt as inputs
     -whichvars is a Python LIST of the variables included in this factor
     where each element of the list is an int in 0,1,...
     -cpt is an ndarray whose dimension is the length of list, and orders the
     dimensions according to the order in which variables appear in whichvars
     """

    def __init__(self, whichvars, cpt):
        self.whichvars = whichvars
        self.tabcpt = TableND(cpt)
        self.cptsize = cpt.size

    def get_cpt_size(self):
        """ foo """
        return self.cptsize

    # def get_var_dim(self, varname):
    #     """ Return the dimension associated with a particular variable """
    #     return self.whichvars == varname

    def get_variable_values(self, index1D):
        """ Return a {list,tuple,array,?} giving the N-D CPT position """
        return self.tabcpt.ind2nd(index1D)

    # def get_specific_variable_value(self, 

def run():
    """ Here's a docstring! """

    # This function will build a graphical model and do some inference on it
    elim_order = [5, 4, 3, 2, 1]

    #http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/...
    #6-438-algorithms-for-inference-fall-2014/lecture-notes/MIT6_438F14_Lec7.pdf
    #
    # Graph structure:
    #  x1---x3---x4
    #  |    |    |
    #  x2---x5----
    #
    # A clique is a subgraph (i.e., a group of nodes from the graph) in which
    # all nodes are connected to all other nodes in the clique. A MAXIMAL
    # clique is a clique that cannot be extended by including one more adjacent
    # vertex. We convert the undirected graph to a factor graph by creating a
    # factor for every clique.

    # Variable domains and potential functions:
    # Let's say for now that all variables are binary.

    # Define first factor: phi12
    wv12 = [1, 2]                        # This factor deals with variables 1,2
    cpt12 = np.array([[1, 0], [0, 1]])   # phi(x1,x2) = 1 if x1==x2
    f12 = Factor(wv12, cpt12)

    # Define second factor: phi13
    wv13 = [1, 3]                        # This factor deals with variables 1,3
    cpt13 = np.array([[1, 0], [0, 1]])   # phi(x1,x3) = 1 if x1==x3
    f13 = Factor(wv13, cpt13)

    # Define third factor: phi25
    wv25 = [2, 5]
    cpt25 = np.array([[1, 0], [0, 1]])
    f25 = Factor(wv25, cpt25)

    # Define fourth factor: phi345
    wv345 = [3, 4, 5]
    cpt345 = np.array( # 3 binary variables ==> 2^3 = 8 possible combinations
            [[[1, 0],
              [0, 0]],
             [[0, 0],
              [0, 1]]])
    f345 = Factor(wv345, cpt345)
    
    factorprod(f12, f13)
   
    # # Test a 3D CPT
    # # It should be that [k,j,i] is a binary-valued index into the array.
    # # For example, [0,1,0] should give back the third element from the "front"
    # # of the order in which this np.array was defined. In this case "3".
    # cptTEST3D = np.array( # 3 binary variables ==> 2^3 = 8 possible combos
    #         [[[1, 2],
    #           [3, 4]],
    #          [[5, 6],
    #           [7, 8]]])
    # print cptTEST3D
    # print cptTEST3D[0,0,0]
    # print cptTEST3D[0,0,1]
    # print cptTEST3D[0,1,0]
    #
    # # Test a 2D CPT
    # cptTEST2D = np.array(
    #         [[1, 2],
    #          [3, 4]])
    # print cptTEST2D
    # print cptTEST2D[0,0]
    # print cptTEST2D[0,1]


if __name__ == "__main__":
    run()