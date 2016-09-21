"""Call various sort algorithms on a list provided at the command line
"""

import argparse
from insertion_sort import insertion_sort
from merge_sort import merge_sort

def main():
    """Call various sorting algorithms on the supplied input values.

    Take in list of numbers, convert them to a list, sort them using
    various algorithms and print the sorted list.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
            help='<Required> Input array to be sorted', required=True)
    args = parser.parse_args()

    # List comprehension conversion to list of ints
    input_list = [int(i) for i in args.input]
    print 'Input list: ' + str(input_list)

    # Insertion Sort
    sorted_list = insertion_sort(input_list)
    print 'Insertion Sort: ' + str(sorted_list)

    # Merge Sort
    sorted_list = merge_sort(input_list)
    print 'Merge Sort: ' + str(sorted_list)

if __name__ == "__main__":
    main()
