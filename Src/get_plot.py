import matplotlib.pyplot as plt
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
