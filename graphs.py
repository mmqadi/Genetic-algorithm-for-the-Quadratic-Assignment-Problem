import numpy as np
import matplotlib.pyplot as plt




def plot(xvalues, yvalues, known_min, xlabel, ylabel, title):
    tries = range(10)
    plt.plot(xvalues, yvalues, color='blue', label='obtained')
    plt.plot(xvalues, known_min, color='red', label='known min')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

fname = "esc32e"
sum_of_best_cost = np.zeros((1000,))

for i in range(0, 10):
    best_cost = np.loadtxt("./iterations/"+fname+"_" + str(i))
    print(len(best_cost))
    sum_of_best_cost += best_cost



average_best_cost = sum_of_best_cost / 10

count2 =0
for i in average_best_cost:
    if i == 2:
        count2 = count2 + 1

per_count = count2 / 1000 * 100
print("count2", per_count)

iterations = range(len(average_best_cost))
known_min = np.asarray([68] * len(average_best_cost))

plot(iterations, average_best_cost, known_min, "iteration number", "average cost", "iteration number vs average cost for GA("+fname+")")

