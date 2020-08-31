from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
xlabels = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')

y = [0.42, 7.44, 18.11, 11.11, 13.36, 0.33, 6.76, 17.5]




# average max value per trial
files = ['nug8', 'nug12', 'nug28', 'tai12a', 'tai20a', 'tai30a', 'esc16a', 'esc32e']

for fname in files:

        sum_of_best_cost = np.zeros((1000,))
        values = []
        for i in range(10):
            data = np.loadtxt("./iterations/"+fname+"_" + str(i))
            data = data / 10

            values.append(np.average(data))



        plt.bar(x, values, color='b')
        plt.xticks(x, xlabels)
        plt.title("average cost per trial ("+fname+")")
        #plt.show()
        plt.savefig('./graph/'+fname+'_average')

