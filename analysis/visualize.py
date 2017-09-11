import matplotlib.pyplot as plt 

class Visualize:

    def plot_results(self, ground_truth, predictions):
        fig = plt.figure()
        fig.suptitle('{} test samples'.format(len(ground_truth)), fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')

        # plt.plot(ground_truth, ground_truth, 'g', label='real', linewidth=0.4)
        for name, values in predictions.iteritems():
            plt.scatter(ground_truth, values, label=name, s=0.5)
        plt.legend(loc='upper left')
        plt.show()