import numpy as np
import matplotlib.pyplot as plt

ranks = np.array([2, 4, 8, 12])

weak_scaling_original_sizes = np.array([1, 2, 4, 8, 12]) * 1024 * 1024 * 1024

weak_serial_time = 2587564717
weak_scaling = np.array(
    [
        3075698770,
        5732301882,
        7663606034,
        7915689555,
    ]
)
weak_size = np.array(
    [
        1041496465,
        2082993914,
        4165986026,
        8331978934,
        12497979500,
    ]
)

strong_scaling_original_size = 12 * 1024 * 1024 * 1024
strong_serial_time = 31922668684
strong_scaling = np.array(
    [
        18616065893,
        11646572930,
        7396770073,
        5439423986,
    ]
)
strong_size = np.array(
    [
        12497979412,
        12497979420,
        12497979436,
        12497979468,
        12497979500,
    ]
)

print("Weak scaling compression ratios")
print(weak_scaling_original_sizes / weak_size)
print("Strong scaling compression ratios")
print(strong_scaling_original_size / strong_size)

plt.plot(ranks, weak_serial_time / weak_scaling)
print("Weak speedup")
print(weak_serial_time / weak_scaling)
plt.xticks(ranks)
plt.ylim(bottom=0)

plt.ylabel("Speedup")
plt.xlabel("Ranks")
plt.title("Weak Scaling Speedup - Uniform Random")
plt.savefig("weak_scaling_uniform.png")

plt.clf()

plt.plot(ranks, strong_serial_time / strong_scaling)
print("Strong speedup")
print(strong_serial_time / strong_scaling)
plt.xticks(ranks)
plt.ylim(bottom=0)

plt.ylabel("Speedup")
plt.xlabel("Ranks")
plt.title("Strong Scaling Speedup - Uniform Random")
plt.savefig("strong_scaling_uniform.png")
