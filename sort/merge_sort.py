"""Implement merge sort for practice.

Note: Comments within the algorithm do not conform to Google Style Guide, but
that's ok because this code is intended for personal learning/study purposes.

"""

import unittest
import numpy.random as nprnd
from insertion_sort import insertion_sort

def merge(A, p, q, r):
    """Merge subroutine.

    Args:
        A: An array whose subarrays are to be sorted.
        p: Index of first element of "left" subarray.
        q: Index of last element of "left" subarray.
        r: Index of last element of "right" subarray.

    Returns:
        A: An array whose subarray A[a,...,r] is sorted.

    Note that the subarrays A[p,...,q] and A[q+1,...,r] are assumed to be
    sorted! This routine will return an incorrect result if these subarrays are
    not sorted because it will only compare the first elements in L and R
    assuming that those are the smallest elements in L and R.

    """

    # Create new arrays to hold the subarrays
    nl = q - p + 1;
    L = [None] * (nl + 1);
    nr = r - q;
    R = [None] * (nr + 1);

    # Copy appropriate elements of A into L and R
    for i in xrange(nl):
        L[i] = A[p + i];
    L[-1] = float("inf")

    for j in xrange(nr):
        R[j] = A[q + 1 + j];
    R[-1] = float("inf")

    # Put elements of L and R back into A in sorted order
    n = nl + nr
    l = r = 0 
    for i in xrange(n):
        if L[l] <= R[r]:
            A[p + i] = L[l]
            l = l + 1
        else:
            A[p + i] = R[r]
            r = r + 1
            
    return A
    

class TestMergeSort(unittest.TestCase):
    """Test components of merge sort.

    Inherets from unittest.TestCase.

    """

    def test_merge_known_input(self):
        """Test the merge subroutine with known, deterministic input.
        """

        A = [1, 3, 5, 2, 4, 6]
        B = merge(A, 0, 2, 5)
        issorted = all(B[i] <= B[i+1] for i in xrange(1,len(B)-1))
        self.assertTrue(issorted)

    def test_merge_rand_input(self):
        """Test the merge subroutine with random input.
        """

        # Test random input: 
        n0 = int(nprnd.randint(low=3, high=8, size=1))
        A0 = list(nprnd.randint(10, size=n0))
        n1 = int(nprnd.randint(low=3, high=8, size=1))
        A1 = list(nprnd.randint(10, size=n1))
        n2 = int(nprnd.randint(low=3, high=8, size=1))
        A2 = list(nprnd.randint(10, size=n2))
        n3 = int(nprnd.randint(low=3, high=8, size=1))
        A3 = list(nprnd.randint(10, size=n3))

        # Sort sub-arrays (recall MERGE assumes sorted sub-arrays as input)
        A1 = insertion_sort(A1)
        A2 = insertion_sort(A2)

        # Put it all together into a list where the middle portions are sorted
        # arrays to be merged. A0 and A3 are random padding to left and right.
        A = A0 + A1 + A2 + A3 

        # Test:
        B = merge(A, n0, n0 + n1 - 1, n0 + n1 + n2 - 1)
        issorted = all(B[i] <= B[i+1] for i in xrange(n0, n0+n1+n2-1))
        self.assertTrue(issorted)

if __name__ == "__main__":
    unittest.main()
