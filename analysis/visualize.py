import matplotlib.pyplot as plt 

class Visualize:

    def plot_results(self, ground_truth, predictions):
        plt.plot(ground_truth, ground_truth, 'g', label='real', linewidth=0.4)
        for name, values in predictions.iteritems():
            plt.scatter(ground_truth, values, label=name, s=0.5)
        plt.legend(loc='upper left')
        plt.show()