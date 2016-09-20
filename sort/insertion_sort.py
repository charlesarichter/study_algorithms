"""Implement the Insertion Sort algorithm for personal practice.

Note: Comments within the algorithm do not conform to Google Style Guide, but
that's ok because this code is intended for personal learning/study purposes.

"""

import argparse

def insertion_sort(input_list):
    """Sort the elements in input_list in ascending numerical order.

    Implements the Insertion Sort algorithm. Works by iterating across the input
    list from left to right, at each step making sure that the current (key)
    element is greater than all elements to the left. If the key is not greater
    than all elements to its left, it shifts those elements to the right by one
    space and inserts the key in the appropriate spot.

    Args:
        input_list: A list of numbers.

    Returns:
        sorted_list: The same elements as input_list, but sorted.

    """

    # Loop over positions in the input list (starting from the second one)
    for j in xrange(1, len(input_list)):

        # Take the element from the specified position and make it the key
        key = input_list[j]

        # We want to compare the key against the elements to the *left* of the
        # key, so we start off with i being one slot to the left of j, and work
        # our way to the left from there
        i = j - 1

        # Loop over elements to the left of the key
        while i >= 0 and input_list[i] > key:

            # If we get this far, input_list[i] is larger than the key, so we
            # need to shift input_list[i] over to the right by one spot to make
            # a space for the key
            input_list[i + 1] = input_list[i]

            # Decrement the counter
            i = i - 1

        # If we get here, we've found where we want to put the key
        # (and it's i+1, not i, because we just decremented i)
        input_list[i + 1] = key

        # Look at the state of the list
        # print input_list
        # raw_input("")

    return input_list


def main():
    """Call insertion_sort on the supplied input values.

    Take in list of numbers, convert them to a list, sort them using
    insertion_sort and print the sorted list.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
            help='<Required> Input array to be sorted', required=True)
    args = parser.parse_args()

    # List comprehension conversion to list of ints
    input_list = [int(i) for i in args.input]

    print 'Sorting the list: ' + str(input_list)
    sorted_list = insertion_sort(input_list)
    print 'Sorted list: ' + str(sorted_list)


if __name__ == "__main__":
    main()
