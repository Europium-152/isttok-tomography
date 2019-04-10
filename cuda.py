import numpy as np
import cupy as cp
import time
import timeit
import matplotlib.pyplot as plt

sizes = [10000]
cpu_times = []
gpu_times = []

for size in sizes:

    matrix = np.random.rand(size, size)
    matrix = np.array(matrix, dtype=np.float32)

    # print(matrix)

    t0 = time.time()
    # inverse = np.linalg.inv(matrix)
    t1 = time.time()
    cpu_times.append(t1-t0)

    print(("cpu inversion: %.16f" % (t1-t0)))

    # print(inverse)
    # print(np.dot(inverse, matrix))

    gpu_matrix = cp.random.rand(size, size, dtype=cp.float32)

    t0 = time.time()
    gpu_invert = cp.linalg.inv(gpu_matrix)
    gpu_invert_oncpu = cp.asnumpy(gpu_invert)
    t1 = time.time()
    gpu_times.append(t1 - t0)

    print(("gpu inversion: %.16f" % (t1-t0)))

    # print(cp.dot(gpu_matrix, gpu_invert))

# plt.figure()
# plt.plot(sizes, cpu_times, label="CPU")
# plt.plot(sizes, gpu_times, label="GPU")
# plt.xlabel("size")
# plt.ylabel(r"time (s)")
# plt.legend()
# plt.show()
