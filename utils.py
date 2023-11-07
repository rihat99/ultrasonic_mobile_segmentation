import torch

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix



def plot_results(results, save_dir):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(results['train_loss'], label='Train loss')
    plt.plot(results['val_loss'], label='Validation loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(results['train_acc'], label='Train accuracy')
    plt.plot(results['val_acc'], label='Validation accuracy')
    plt.legend()
    plt.savefig(save_dir + '/LossAccuracy.png')

def plot_several_results(all_results, variable_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    i = 0
    for key, value in all_results.items():
        if i == 0: color = 'blue'
        elif i == 1: color = 'orange'
        elif i == 2: color = 'green'
        else : color = 'red'

        ax1.plot(value['train_loss'], label=f'{variable_name}={key}, train loss', color=color, )
        ax1.plot(value['val_loss'], label=f'{variable_name}={key}, val loss', color=color, linestyle='dashed')
        ax1.legend()

        ax2.plot(value['train_acc'], label=f'{variable_name}={key}, train acc', color=color, )
        ax2.plot(value['val_acc'], label=f'{variable_name}={key}, val acc', color=color, linestyle='dashed')
        ax2.legend()

        i+=1

    plt.show()


def plot_confussion_matrix(save_dir, model, loader, device, classes, normalize=False):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # add numbers in the boxes
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.savefig(save_dir + '/ConfusionMatrix.png')


def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model