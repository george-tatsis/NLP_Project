import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from __main__ import *
sns.set()

def get_plot(history, architecture, date_time, hidden_layers, epochs, labels, metric, title, ylabel):

    epochs = range(1,epochs+1)

    try:
      os.mkdir("plots/{}_{}".format(architecture,hidden_layers))
    except:
      pass
    
    plt.figure(figsize=(15,10))
    for ind in range(len(history)):
        val = history[ind].history[metric]
        plt.plot(epochs, val, label=labels[ind])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('plots/{}_{}/{}_{}.png'.format(architecture,hidden_layers,title,date_time))


def time_plot(simplernn, lstm, gru, title='Time_bar'):
    adam = [simplernn['adam'], lstm['adam'], gru['adam']]
    sgd = [simplernn['sgd'], lstm['sgd'], gru['sgd']]
    rmsprop = [simplernn['rmsprop'], lstm['rmsprop'], gru['rmsprop']]

    fig, ax = plt.subplots(figsize=(10, 6))
    num_class = 3
    x = np.arange(num_class)
    wid = 0.2

    p1 = ax.bar(x, adam, width=wid, color='orangered', bottom=0)
    p2 = ax.bar(x + wid, sgd, width=wid, color='slateblue', bottom=0)
    p3 = ax.bar(x + 2 * wid, rmsprop, width=wid, color='chartreuse', bottom=0)

    ax.set_title('Plot')

    # Plot labels
    plt.title("Performance Graph")
    plt.xlabel("Algorithms")
    plt.ylabel("Time")
    plt.xticks(x + wid, ['SimpleRNN', 'LSTM', 'GRU'])

    # Plot legends
    red_patch = mpatches.Patch(color='orangered', label='Adam Optimizer')
    blue_patch = mpatches.Patch(color='slateblue', label='SGD Optimizer')
    green_patch = mpatches.Patch(color='chartreuse', label='RMSProp Optimizer')
    plt.legend(bbox_to_anchor=(1, 1), handles=[red_patch, blue_patch, green_patch])

    plt.savefig('plots/{}_{}.png'.format(title,date_time))
