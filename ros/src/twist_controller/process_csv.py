#!/usr/bin/env python
# Ideas from https://pythonprogramming.net/loading-file-data-matplotlib-tutorial/
import matplotlib.pyplot as plt
import csv

def plot_file(file_name, title, legend1, legend2):
    '''
    Plots csv data
    '''
    x = []
    y = []
    z = []
    with open(file_name,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in plots:
            count += 1
            if count == 1:
                continue
            x.append(count)
            y.append(float(row[0]))
            z.append(float(row[1]))

    plt.gca().set_color_cycle(['red', 'blue', 'green', 'yellow'])

    plt.plot(x,y, label=legend1)
    plt.plot(x,z, label=legend2)

    plt.title(title)
    plt.legend([legend1, legend2], loc='upper left')
    plt.savefig('{}.png'.format(title))

# Process CSV files
plot_file('throttles.csv', 'throttles', 'actual', 'proposed')
plot_file('steers.csv', 'steers', 'actual', 'proposed')
plot_file('brakes.csv', 'brakes', 'actual', 'proposed')


