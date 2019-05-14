import numpy as np
import matplotlib.pyplot as plt
filepath = input('Enter file path you want to plot: ')
title = input('Enter plot title: ')
xaxis = input('Enter x- axis name: ')
yaxis = input('Enter y- axis name: ')

a = unpickle('{}'.format(filepath))

plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
plt.title(title)
plt.xlabel(xaxis)
plt.ylabel(yaxis)
plt.plot(a)

plt.savefig('{}.png'.format(filepath))
