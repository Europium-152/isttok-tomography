import shutil
import os
import sys
from distutils.util import strtobool
import numpy as np

valid_projections = [
    "line-approximation.npy",
    "complex-view-cone.npy",
    "complex-view-cone-128.npy"
]

# Module directory. Use to fetch the projection files
this_directory = os.path.dirname(__file__) + '\\projections\\'


def select_projection(file_to_copy):

    if os.path.exists("../projections.npy"):
        answer = strtobool(input("Do you wish to overwrite the existing projection configuration? (y/n)"))
        if answer:
            shutil.copy(file_to_copy, "../projections.npy")
            print("Done")
        else:
            print("Aborted")
    else:
        shutil.copy(file_to_copy, "../projections.npy")
        print("Done")


def load_projection(projection_file_name):

    try:
        return np.load(this_directory + projection_file_name)
    except IOError:
        raise IOError("'%s' is not a valid projection. Select one from:\n" % (this_directory + projection_file_name)
                      + str(valid_projections))


if __name__ == "__main__":
    if (len(sys.argv) < 2) or (not sys.argv[1] in valid_projections):
        print("\nFunction needs at least one argument: $ projection-selector.py <desired_projection>\n")
        print("Available projections are:\n")
        for projection in valid_projections:
            print(projection)
    else:
        select_projection(sys.argv[1])

