import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import time
import random
import torch.optim as optim
import itertools
import argparse
import wandb
import csv
from datetime import datetime
import timm
from functools import partial
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import joblib
from focal_loss.focal_loss import FocalLoss

def check_positive(value):
    """
    Custom argparse type for positive integers.

    Args:
        value: The input value to be checked.

    Returns:
        int: The positive integer value.

    Raises:
        argparse.ArgumentTypeError: If the input value is not a positive integer.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        tuple: Model name, verbosity flag, number of trials, and image size.
    """
    VALID_MODELS = ['MaxViT', 'ConvNeXt']
    VALID_SIZES = [224, 384, 512]

    # Create an ArgumentParser object and define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=VALID_MODELS, default='ResNet')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--trials', type=check_positive, default=0)
    parser.add_argument('--image_size', choices=VALID_SIZES, type=check_positive, default=512)

    args = parser.parse_args()

    model_name = args.model
    verbose = args.verbose
    num_trials = args.trials
    img_size = args.image_size

    return model_name, verbose, num_trials, img_size

def plot_confusion_matrix(y_true, y_pred, unique_labels, path, name):
    """
    Plot and save a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        unique_labels (list): List of unique labels/classes.
        path (str): Path to save the plot.
        name (str): Name of the model.
    """

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
    """
    Plot and save loss curves.

    Args:
        epochs: List of epochs.
        train_losses: List of training losses.
        val_losses: List of validation losses.
        path (str): Path to save the plot.
        name (str): Name of the model.
    """

    plt.figure(figsize=(9,7),dpi=150)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Error Loss per Epoch: "+name)

    loss_df = pd.DataFrame({
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })

    # Plot the training and validation loss
    sns.lineplot(data=loss_df, dashes=False, errorbar=None)
    
    plt.legend()    
    plt.savefig(path+name+'_Training_Loss.png')

def plot_roc_auc(epochs, train_roc_auc_scores, val_roc_auc_scores, path, name):
    """
    Plot and save ROC AUC during training.

    Args:
        epochs: List of epochs.
        train_roc_auc_scores: List of training ROC AUC scores.
        val_roc_auc_scores: List of validation ROC AUC scores.
        path (str): Path to save the plot.
        name (str): Name of the model.
    """
    
    plt.figure(figsize=(9,7),dpi=150)
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC per Epoch: "+name)

    roc_auc_df = pd.DataFrame({
        'Training': train_roc_auc_scores,
        'Validation': val_roc_auc_scores
    })

    sns.lineplot(data=roc_auc_df, dashes=False, errorbar=None)

    plt.legend() 
    plt.savefig(path+name+'_ROC_AUC.png')

def plot_bal_acc(epochs, train_bal_acc_scores, val_bal_acc_scores, path, name):
    """
    Plot and save balanced accuracy during training.

    Args:
        epochs: List of epochs.
        train_bal_acc_scores: List of training balanced accuracy scores.
        val_bal_acc_scores: List of validation balanced accuracy scores.
        path (str): Path to save the plot.
        name (str): Name of the model.
    """
    
    plt.figure(figsize=(9,7),dpi=150)
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy per Epoch: "+name)

    # Create a DataFrame from the lists
    acc_df = pd.DataFrame({
        'Training': train_bal_acc_scores,
        'Validation': val_bal_acc_scores
    })

    sns.lineplot(data=acc_df, dashes=False, errorbar=None)

    plt.legend() 
    plt.savefig(path+name+'_Balanced_Accuracy.png')

def plot_roc_curve(y_true, y_score, num_classes, class_labels, path, name):
    """
    Plot and save ROC curves.

    Args:
        y_true: True labels.
        y_score: Predicted scores/probabilities.
        num_classes (int): Number of classes.
        class_labels: List of class labels.
        path (str): Path to save the plot.
        name (str): Name of the model.
    """

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve
    with sns.axes_style("white"):
        plt.figure(figsize=(9,7),dpi=150)
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'.format(class_labels[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic: '+name)
        plt.legend(loc="lower right")

        plt.savefig(path+name+'_ROC_Curve.png')\
    
def save_parameters_to_csv(filepath, parameters, name):
    """
    Save parameters and their values to a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        parameters (dict): Dictionary of parameter names and values.
        name (str): Name of the model.
    """
    
    # Check if the file exists
    file_exists = os.path.exists(filepath)

    # Open the file in append mode
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Date', 'Time', 'Model_Name'] + list(parameters.keys()))

        # Write headers if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Get current date and time
        now = datetime.now()
        current_date = now.strftime('%d-%m-%Y')
        current_time = now.strftime('%H:%M:%S')

        # Append a new entry with date, time, and parameter values
        parameters_with_datetime = {'Date': current_date, 'Time': current_time, 'Model_Name':name, **parameters}
        writer.writerow(parameters_with_datetime)

import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from data import *
from train import *
sns.set_theme()
sns.set(font_scale=1.4)

import gc
gc.collect()