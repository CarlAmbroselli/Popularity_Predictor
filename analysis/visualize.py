import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

    def plot_roc(self, ground_truth, predictions):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
