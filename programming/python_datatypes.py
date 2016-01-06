import numpy as np

def mainFunction():

    # Three Python native datatypes:
    # 1) List: [] container that can be modified
    # 2) Tuple: () just like a list, but immutable once created,  and faster
    # 3) Dictionary: {} key-value pairs

    # List
    li = ["a", "b", "c"]
    print li[0:2]

    # Tuple
    tu = ("a", "b", "c")

    # Dictionary
    di = {"key1": "value1", "key2": "value2", "key3": "value3"}

    # Numpy "types": https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
    # 1) Array: n-dimensional array, which is the fundamental numpy type
    # 1a) Matrix: sub-class of array for 2D, with some linear algebra operators defined

    # Array creation: http://docs.scipy.org/doc/numpy-1.10.0/reference/
    #     routines.array-creation.html#routines-array-creation
    ar = np.array([[1, 2, 3], [4, 5, 6]], np.float) 
    vec = np.array([[1], [2], [3]], np.float)
    print ar
    print vec

    # Array multiplication
    res = np.dot(ar,vec)
    print res

    # Arrays need compatible dimensions to multiply
    # res = np.dot(vec,ar) # won't work

    # Creating a numpy array from an existing python list or tuple
    li = [0, 1, 2, 3]
    ar = np.array([li])
    print ar
    print ar.T

if __name__ == "__main__":
    mainFunction()


