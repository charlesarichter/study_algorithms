""" N-dimensional tabular data container:
    Enables conversion between 1D and ND indexing which is useful for
    representing conditional probability tables (CPTs), lookup table
    approximations of learned models, etc. """

import numpy as np

class TableND(object):
    """ TableND class """

    def __init__(self, data):

        """ Data is stored as a 1D array """ 
        self.data = data.flatten()

        """ Store the cardinality of each variable (how many different values
        could it take on) """
        self.card = data.shape

        """ Store the number of dimensions """
        self.ndims = len(data.shape)

        # Compute a cumulative product of dimension sizes
        self.cprod = np.ones(self.ndims + 1)
        for i in xrange(self.ndims):
            self.cprod[self.ndims - i - 1] = self.card[self.ndims - i - 1] * \
                    self.cprod[self.ndims - i]

    def get1D(self, index1D):
        """ Get an element of the table """
        return self.data[index1D]

    def getND(self, indexND):
        """ Get an element of the table """
        return self.data[self.nd2ind(indexND)]

    def set1D(self, index1D, val):
        """ Set an element of the table """
        self.data[index1D] = val

    def setND(self, indexND, val):
        """ Set an element of the table """
        self.data[self.nd2ind(indexND)] = val
    
    def nd2ind(self, nd):
        """ Convert ND index (python list, numpy array or similar) to 1D
        index """

        ind = 0
        for i in xrange(self.ndims):

            if nd[self.ndims - i - 1] >= self.card[self.ndims - i - 1]:
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

    """ Test out TableND class """
    cpt = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tablend = TableND(cpt) 

    """ Test indexing """
    for i in xrange(cpt.size):
        # print i
        nd = tablend.ind2nd(i)
        # print nd
        ind = tablend.nd2ind(nd)
        if ind != i:
            print "oops!"

    """ Test Setters and Getters """
    a = np.array([1,0,2])
    print tablend.getND(a)
    print tablend.get1D(0)
    tablend.setND(a,64)
    print tablend.getND(a)
    tablend.set1D(6,32)
    print tablend.data
