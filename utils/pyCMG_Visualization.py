'''
Author: Yunan Li
Date: 2024/02/17
Goal: This file contains class and functions to demonstarte and visualize results.
Class:  1. rst_vi
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
# import plotly.express as px

import seaborn as sns

class rst_viz():
    def __init__(self):
        super().__init__()


    def visualize_corr(self, df):

        # Compute the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(8, 6))

        # Change fontsize
        sns.set(font_scale=1.5)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, annot=False)
        

    def plot_image_comparison(image, ae_pred, sample_idx=False, time_idx=False, cmap=plt.cm.gray, cbar_label="Land surface uplift, ft"):
        # from mpl_toolkits.axes_grid1 import make_axes_locatable

        ##### Check the index: sample_idx and time_idx start from 1 #####
        if sample_idx is not False and sample_idx < 1:
            print('The sample_idx should start from 1 .....')
        if time_idx is not False and time_idx < 1:
            print('The time_idx should start from 1 .....')


        if sample_idx and time_idx:
            raw_image = image[int(sample_idx-1), :, :, int(time_idx-1)]
            pred_image = ae_pred[int(sample_idx-1), :, :, int(time_idx-1)]
            title = f"The {int(sample_idx)} sample after {int(time_idx)} years of injection"
        elif sample_idx:
            raw_image = image[int(sample_idx-1), :, :]
            pred_image = ae_pred[int(sample_idx-1), :, :]
            title = f"The {int(sample_idx)} sample"
        elif time_idx:
            raw_image = image[:, :, int(time_idx-1)]
            pred_image = ae_pred[:, :, int(time_idx-1)]
            title = f"Query at time after {int(time_idx)} years of injection"
        else:
            raw_image = image
            pred_image = ae_pred
            title = ""

        abs_diff = np.abs(raw_image - pred_image)
        max_val = max(np.max(raw_image), np.max(pred_image))
        max_val_diff = np.max(abs_diff)

        plt.figure(figsize = (18,5))

        plt.subplot(1, 3, 1)
        ax1 = plt.gca()
        if cmap:
            plt.imshow(raw_image,cmap = cmap,vmin=0,vmax=max_val)
        else:
            plt.imshow(raw_image,vmin=0,vmax=max_val)
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.title('Raw image', fontsize=14)
        ax1.axes.get_xaxis().set_ticks([])
        ax1.axes.get_yaxis().set_ticks([])
        ### Set color bar ###
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)


        plt.subplot(1, 3, 2)
        ax2 = plt.gca()
        if cmap:
            plt.imshow(pred_image,cmap = cmap,vmin=0,vmax=max_val)
        else:
            plt.imshow(pred_image,vmin=0,vmax=max_val)
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.title('Reconstructed image', fontsize=14)
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)


        plt.subplot(1, 3, 3)
        ax3 = plt.gca()
        if cmap:
            plt.imshow(abs_diff,cmap = cmap,vmin=0,vmax=max_val)
        else:
            plt.imshow(abs_diff,vmin=0,vmax=max_val_diff)
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.title('Absoluate difference', fontsize=14)
        ax3.axes.get_xaxis().set_ticks([])
        ax3.axes.get_yaxis().set_ticks([])
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.suptitle(title, fontsize=18)
        plt.show()

