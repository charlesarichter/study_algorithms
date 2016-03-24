""" Here's a docstring! """

import numpy as np
from sets import Set
from tablend import TableND

def issorted(ar):
    """ Check if an array is sorted in ascending numerical order """
    return all(ar[i] <= ar[i+1] for i in xrange(len(ar)-1))

def getprodmembers(wv1,vals1,wv2,vals2):
    """ Compute the variables involved in the product of factor1 and factor2
    along with their domains 
   
    Input: Which variables involved in both factors, and their values
    Output: The union of the two sets of variables, and their values
    
    """

    output_set = Set([])
    output_vars = []
    output_vals = []

    for i in xrange(len(wv1)):
        
        var = wv1[i]

        if var not in output_set:

            # Add to the set
            output_set.add(var)
            
            # Also add to an (ordered list)
            output_vars.append(var)
            output_vals.append(vals1[i])

    for j in xrange(len(wv2)):
        
        var = wv2[j]

        if var not in output_set:

            # Add to the set
            output_set.add(var)
            
            # Also add to an (ordered list)
            output_vars.append(var)
            output_vals.append(vals2[j])

    return (output_vars,output_vals)

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
    print "  Factor product output variables: ", output_vars
    print "--FIXME: Are the output variables sorted??"

    """ Get the domain size of the output variables """
    return_domains = 2 * np.ones(len(output_vars))
    print "--FIXME: Hardcoded domains for now: ", return_domains

    """ Make a new empty factor """
    f_return = Factor(list(output_vars), np.zeros(return_domains))

    """ Get a list of variables that overlap between the two factors """
    overlap_vars = list(set(factor1.whichvars) & set(factor2.whichvars))
    # print overlap_vars

    """ For each overlap variable, record which dim of each cpt is involved """
    overlap_dims = []
    for i in xrange(len(overlap_vars)):
        d1 = factor1.whichvars.index(overlap_vars[i])
        d2 = factor2.whichvars.index(overlap_vars[i])
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

            # print ""
            # print "Which variables for factor 1: " 
            # print factor1.whichvars
            # print "Values of factor 1 variables: "
            # print variable_values1
            # print "Which variables for factor 2: " 
            # print factor2.whichvars 
            # print "Values of factor 2 variables: "
            # print variable_values2

            """ Get the variables and assignments involved """
            prodmembers = getprodmembers(factor1.whichvars,variable_values1,
                factor2.whichvars,variable_values2)
            prodvars = prodmembers[0]
            prodvals = prodmembers[1]
            # print "Output factor variables: ", prodvars
            # print "Output factor values: ", prodvals

            """ Get the activations for the two parent factors """
            a1 = factor1.get_activationND(variable_values1)
            a2 = factor2.get_activationND(variable_values2)

            """ Multiply activations and store result in the new factor """
            f_return.set_activationND(prodvals,a1*a2)

            # raw_input("pause")

    # print f_return.tabcpt.data
    return f_return

def factormarginal(factor,whichvar):
    """ Sum over whichvar and output a new factor """

    # print factor.whichvars
    # print factor.whichvars.index(whichvar)
    output_vars = list(factor.whichvars)
    # print output_vars
    output_vars = [x for x in output_vars if x != whichvar]
    # print output_vars

    indmarginal = factor.whichvars.index(whichvar)
    # print indmarginal

    return_domains = 2 * np.ones(len(output_vars))
    print "--FIXME: Hardcoded domains for now: ", return_domains

    f_return = Factor(output_vars, np.zeros(return_domains))

    """ Iterate over all entries in the input factor """
    for i in xrange(factor.get_cpt_size()):
        variable_values = factor.get_variable_values(i)
        # print variable_values

        """ Get the index into the output factor """
        # print indmarginal
        var_values_list = list(variable_values)
        del var_values_list[indmarginal]
        # print var_values_list

        """ Increment output factor """
        act_return = f_return.get_activationND(var_values_list)
        act = factor.get_activationND(variable_values)
        f_return.set_activationND(var_values_list,act_return + act)

    # print f_return.tabcpt.data
    return f_return

def factornormalize(factor):
    """ Normalize values in CPT of a factor and output a new factor """


class Factor(object):
    """ Factor class takes whichvars and cpt as inputs
     -whichvars is a Python LIST of the variables included in this factor
     where each element of the list is an int in 0,1,...
     -cpt is an ndarray whose dimension is the length of list, and orders the
     dimensions according to the order in which variables appear in whichvars
     """

    def __init__(self, whichvars, cpt):
        """ Set the 'names' of variables in this factor """
        self.whichvars = whichvars
        self.tabcpt = TableND(cpt)
        self.cptsize = cpt.size

        """ Check to make sure whichvars is in ascending numerical order """
        if not issorted(self.whichvars):
            print "WARNING: whichvars is not sorted! whichvars = ", whichvars
            exit()

    def __str__(self):
        """ String representation of this factor """
        return 'Factor with variables: ' + str(self.whichvars)

    def normalize(self):
        """ Normalize the values in the CPT """
        self.tabcpt.data = [float(x)/sum(self.tabcpt.data) for x in \
                self.tabcpt.data]

    def includes_var(self, var):
        """ Return whether this factor includes the specified variable """
        return var in self.whichvars

    def get_cpt_size(self):
        """ Return number of elements in the CPT """
        return self.cptsize

    def get_variable_values(self, index1D):
        """ Return a {list,tuple,array,?} giving the N-D CPT position """
        return self.tabcpt.ind2nd(index1D)

    def get_activationName(self, name):
        """ Get an element of the CPT, querying by name """
        """ Fill me in! """

    def get_activation1D(self, index1D):
        """ Get an element of the CPT """
        return self.tabcpt.get1D(index1D) 

    def get_activationND(self, indexND):
        """ Get an element of the CPT """
        return self.tabcpt.getND(indexND) 

    def set_activationName(self, name, val):
        """ Set an element of the CPT, querying by name """
        """ Fill me in! """

    def set_activation1D(self, index1D, val):
        """ Set an element of the CPT """
        self.tabcpt.set1D(index1D,val)

    def set_activationND(self, indexND, val):
        """ Set an element of the CPT """
        self.tabcpt.setND(indexND,val)

def getinvolvedfactors(factors, e):
    """ Return list of factors that involve variable e """
    involved_factors = []
    for f in factors:
        if f.includes_var(e):
            involved_factors.append(f)
    return involved_factors

def undirectedelim(factors, elim_order):
    """ Takes in a set of factors and an elimination order and computes the
    marginal probability of the final variable in the elimination order """

    for e in elim_order:
        print "Eliminating variable: ", e
        
        involved_factors = getinvolvedfactors(factors, e)

        if len(involved_factors) == 0:
            print "  Number of involved factors = 0"
            print "WARNING: Not sure what to do here..."

        elif len(involved_factors) == 1:
            print "  Number of involved factors = 1"
            f = involved_factors[0]

        elif len(involved_factors) == 2:
            print "  Number of involved factors = 2"
            f = factorprod(involved_factors[0],involved_factors[1])
        else:
            print "  Number of involved factors > 2"
            f = factorprod(involved_factors[0],involved_factors[1])
            for i in xrange(2,len(involved_factors)):
                f = factorprod(f,involved_factors[i])

        """ Remove the involved factors from the queue """
        for f_used in involved_factors:
            # print f.whichvars
            factors.remove(f_used)

        """ Now that we have computed the factor product sum out e """
        f = factormarginal(f, e) 

        """ Add the newly created message to the queue """
        factors.append(f)

        # print "List of factors now has length: ", len(factors)
        # print "and add the newly created message to the queue"
        
            
        # raw_input("pause")

    # print "Size of remaining factor list: ", len(factors)
    f = factors[0] 
    f.normalize()
    # print f.tabcpt.data
    return f
 
def run():
    """ Here's a docstring! """

    # List of factors
    factors = []

    # This function will build a graphical model and do some inference on it
    elim_order = [5, 4, 3, 2]
    # elim_order = [4, 5, 2, 3]

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
    # cpt12 = np.array([[1, 0], [0, 1]])   # phi(x1,x2) = 1 if x1==x2
    cpt12 = np.array([[1, 0], [0, 2]])   # phi(x1,x2) = 1 if x1==x2
    f12 = Factor(wv12, cpt12)
    factors.append(f12)

    # Define second factor: phi13
    wv13 = [1, 3]                        # This factor deals with variables 1,3
    # cpt13 = np.array([[1, 0], [0, 1]])   # phi(x1,x3) = 1 if x1==x3
    cpt13 = np.array([[1, 0], [0, .75]])   # phi(x1,x3) = 1 if x1==x3
    f13 = Factor(wv13, cpt13)
    factors.append(f13)

    # Define third factor: phi25
    wv25 = [2, 5]
    cpt25 = np.array([[1, 0], [0, 1]])
    f25 = Factor(wv25, cpt25)
    factors.append(f25)

    # Define fourth factor: phi345
    wv345 = [3, 4, 5]
    cpt345 = np.array( # 3 binary variables ==> 2^3 = 8 possible combinations
            [[[1, 0],
              [0, 0]],
             [[0, 0],
              [0, 1]]])
    f345 = Factor(wv345, cpt345)
    factors.append(f345)

    """ Try undirected elimination """
    f = undirectedelim(factors,elim_order)
    print f.tabcpt.data

    # """ Try a factor product """
    # f2345 = factorprod(f25, f345)
    #
    # """ Try a factor marginalization """
    # f234 = factormarginal(f2345, 5)
    #
    # """ Try a factor marginalization """
    # f23 = factormarginal(f234, 4)
    #
    # """ Try a factor product """
    # f123 = factorprod(f23, f13)
    #
    # """ Try a factor marginalization """
    # m12 = factormarginal(f123, 3)
    #
    # """ Try a factor product """
    # mm12 = factorprod(m12, f12)
    #
    # """ Try a factor marginalization """
    # f1 = factormarginal(mm12, 2)
    # print "-------------------------"
    # print "Result: ", f1.tabcpt.data

    # # Define a factor with variables "out of order" to test if it gets caught
    # wv21 = [2, 1]
    # cpt21 = np.array([[1, 0], [0, 1]])
    # f21 = Factor(wv21, cpt21)

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
