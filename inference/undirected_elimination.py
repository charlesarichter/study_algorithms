""" Here's a docstring! """

import numpy as np
from sets import Set

def factorprod(factor1, factor2):
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

    # out_vars = list(output_set)

    # Now loop over the output variables to compute products

# # Normally, here we would do something like:
#     for var1 in xrange(1):
#         for var2 in xrange(1):
#             for var3 in xrange(1):
#                # Compute product here
# But we can't do this, because we don't know which vars to loop over...

    # for x in np.nditer(factor2.cpt):
    #     print x 

    print factor1.get_cpt_size()
    print factor1.get_cpt_shape()

    for i in xrange(factor1.get_cpt_size()):
        for j in xrange(factor2.get_cpt_size()):
            print "i = ", i, ", j = ", j 



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
        # self.vardims = vardims
        self.cpt = cpt
        # self.size = self.cpt.size

    def get_cpt_size(self):
        """ foo """
        return self.cpt.size

    def get_cpt_shape(self):
        """ foo """
        return self.cpt.shape

    def ind_to_nd(self, whichelement):
        """ foo """
        int 


    def get_cpt_element(self, whichelement):
        """ foo """
        


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
