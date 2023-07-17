import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import numpy as np
import time
import random
import torch.optim as optim
import itertools
import argparse
import wandb

def parse_arguments():
# Define the accepted values for the arguments
    VALID_MODELS = ['Swin', 'ViT', 'ResNet', 'ConvNeXt']
    VALID_TRANSFER_LEARNING = ['single', 'multi']

    # Create an ArgumentParser object and define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=VALID_MODELS, default='ResNet')
    parser.add_argument('--transferlearning', choices=VALID_TRANSFER_LEARNING, default='single')
    parser.add_argument('--verbose', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    # Save the arguments as variables
    model_name = args.model
    transfer_learning = args.transferlearning
    verbose = args.verbose

    return model_name, transfer_learning, verbose

def plot_confusion_matrix(y_true, y_pred, unique_labels, path, name):
    
    plt.figure(figsize=(9,7),dpi=150)
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_pred), index=unique_labels, columns=unique_labels)
    print('\nConfusion Matrix:\n',conf_mat,'\n',sep='')
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', linewidth=.5, square=True)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title("Confusion Matrix: "+name)
    plt.show()
    plt.savefig(path+name+'_Confusion_Matrix.png')

def plot_loss(epochs, train_losses, val_losses, path, name):
    
    plt.figure(figsize=(9,7),dpi=150)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Error Loss per Epoch: "+name)

    # Create a DataFrame from the lists
    loss_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })

    # Plot the training and validation loss
    sns.lineplot(data=loss_df, x='Epoch', y='Training Loss', ci=None)
    sns.lineplot(data=loss_df, x='Epoch', y='Validation Loss', ci=None)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    
    plt.savefig(path+name+'_Training_Loss.png')

def plot_roc_auc(epochs, roc_auc_scores, path, name):
    plt.figure(figsize=(9,7),dpi=150)
    plt.xlabel("Epochs")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC per Epoch: "+name)

    # Create a DataFrame from the lists
    loss_df = pd.DataFrame({
        'Epoch': epochs,
        'ROC_AUC': roc_auc_scores
    })

    sns.lineplot(data=loss_df, x='Epoch', y='ROC_AUC', ci=None)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    plt.savefig(path+name+'_ROC_AUC.png')

def plot_roc_curve(y_true, y_score, num_classes, class_labels, path, name):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve
    plt.figure(figsize=(9,7),dpi=150)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(class_labels[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic: '+name)
    plt.legend(loc="lower right")

    plt.savefig(path+name+'_ROC_Curve.png')

import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data import *
from train import *