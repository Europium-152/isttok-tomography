from __future__ import print_function
import shutil
import os
import sys
from distutils.util import strtobool

valid_projections = [
    "line-approximation.npy",
    "complex-view-cone.npy"
]


def select_projection(file_to_copy):

    if os.path.exists("../projections.npy"):
        answer = strtobool(raw_input("Do you wish to overwrite the existing projection configuration? (y/n)"))
        if answer:
            shutil.copy(file_to_copy, "../projections.npy")
            print("Done")
        else:
            print("Aborted")
    else:
        shutil.copy(file_to_copy, "../projections.npy")
        print("Done")


if __name__ == "__main__":
    if (len(sys.argv) < 2) or (not sys.argv[1] in valid_projections):
        print("\nFunction needs at least one argument: $ projection-selector.py <desired_projection>\n")
        print("Available projections are:\n")
        for projection in valid_projections:
            print(projection)
    else:
        select_projection(sys.argv[1])

