from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generates all performance measures

def metrics(true_labels, pred_labels, plot_title, work='test'):
    cf_matrix = confusion_matrix(true_labels, pred_labels)
    TP, TN, FP, FN = cf_matrix[1][1], cf_matrix[0][0], cf_matrix[0][1], cf_matrix[1][0]
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1_score = 2*Recall*Precision/(Recall+Precision)
    Accuracy = (TP+TN)/true_labels.shape[0]
    
    if work == 'test':
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

        ax.set_title(plot_title)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['0','1'])
        ax.yaxis.set_ticklabels(['0','1'])

        plt.savefig('cf_matrix_plots/'+plot_title+'.png')

        return F1_score, Accuracy

    else:
        return F1_score, Accuracy, TP, TN, FP, FN

def plot_val_cf_matrix(true_labels, pred_labels, plot_title, TP, TN, FP, FN):
    cf_matrix = confusion_matrix(true_labels, pred_labels)
    cf_matrix[1][1], cf_matrix[0][0], cf_matrix[0][1], cf_matrix[1][0] = TP, TN, FP, FN
    
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title(plot_title)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

    plt.savefig('cf_matrix_plots/'+plot_title+'_val.png')
