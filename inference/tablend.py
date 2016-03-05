""" N-dimensional tabular data container:
    Enables conversion between 1D and ND indexing which is useful for
    representing conditional probability tables (CPTs), lookup table
    approximations of learned models, etc. """

import numpy as np

class TableND(object):
    """ TableND class """

    def __init__(self, data):

        # Set class data variable to hold the actual data
        self.data = data
        self.dims = data.shape
        self.ndims = len(data.shape)

        # Compute a cumulative product of dimension sizes
        self.cprod = np.ones(self.ndims + 1)
        for i in xrange(self.ndims):
            self.cprod[self.ndims - i - 1] = self.dims[self.ndims - i - 1] * \
                    self.cprod[self.ndims - i]

    def nd2ind(self, nd):
        """ Convert ND index (python list, numpy array or similar) to 1D
        index """

        ind = 0
        for i in xrange(self.ndims):

            if nd[self.ndims - i - 1] >= self.dims[self.ndims - i - 1]:
                print "Tried to access an ND point out of bounds!"
                return -1

            ind += self.cprod[self.ndims - i] * nd[self.ndims - i - 1]
        return ind

    def ind2nd(self, ind):
        """ Convert 1D index to ND index """

        index = ind
        nd = np.zeros(self.ndims)
        for i in xrange(self.ndims):
            nd[i] = index // self.cprod[i + 1] # // is "floor division"
            index -= nd[i] * self.cprod[i + 1]
        return nd

if __name__ == "__main__":

    # Test out TableND class
    cpt = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tablend = TableND(cpt) 
    for i in xrange(cpt.size):
        print i
        nd = tablend.ind2nd(i)
        print nd
        ind = tablend.nd2ind(nd)
        if ind != i:
            print "oops!"
