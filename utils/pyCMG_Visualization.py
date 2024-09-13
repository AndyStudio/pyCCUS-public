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
from sklearn.decomposition import PCA, IncrementalPCA
import seaborn as sns
from scipy.stats import norm

import sys
# append the path of the parent directory
sys.path.append("..")
# import method from sibling module
from utils.pyCMG_Results import pycmgpostcalc


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

    ################################################
    ##### plt.imshow multiple imgs in subplots #####
    ################################################
    def plot_all_in_npy(self, cidx, tidx, prop, colormap=None, savefig=None):
        """
        Goal: plot 6 images all together.

        Inputs
        1. cidx: int. index for the caseid dimension
        2. tidx: int. index for the time dimension
        3. prop: dict. the names of keys have to be the same as the arr names already loaded
        4. colormap: plt color maps. e.g. colormap = plt.cm.gray_r
        5. savefig: str. Path to save the fig. e.g. '../data/figs/plot1.png'
        Outputs
        None
        """

        postcalc = pycmgpostcalc()
        postcalc.rescaleX_num_grids = 200
        postcalc.rescaleY_num_grids = 200
    
    
        caseidx_select = cidx
        timeidx_select = tidx


        plt.figure(figsize = (16,9))


        for ii, pp in enumerate(prop):

            arr = globals()[pp]
            rescale_arr = postcalc.rescale_ijk2xyz(X_2d=arr[caseidx_select, timeidx_select, :, :])

            plt.subplot(2, 3, int(ii+1))
            ax1 = plt.gca()
            if colormap:
                plt.imshow(rescale_arr,cmap = colormap,vmin=np.min(rescale_arr),vmax=np.max(rescale_arr))
            else:
                plt.imshow(rescale_arr,vmin=np.min(rescale_arr),vmax=np.max(rescale_arr))

            plt.xlabel('X coordinate', fontsize=14)
            plt.ylabel('Y coordinate', fontsize=14)
            plt.title(prop[pp], fontsize=14)
            ax1.axes.get_xaxis().set_ticks([])
            ax1.axes.get_yaxis().set_ticks([])

            ### Set color bar ###
            cbar = plt.colorbar(shrink=0.7)
            # cbar.set_label(prop[pp], fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        # plt.suptitle(f'wellopt3 case{dfsimstatus.caseid.to_numpy()[caseidx_select]} at {time_query[timeidx_select]}yr', fontsize=20)
        plt.suptitle(f'Case index of {caseidx_select} at time index of {timeidx_select}', fontsize=20)
        # plt.tight_layout()
        if savefig:
            plt.savefig(savefig,dpi=300,bbox_inches='tight')
        plt.show()



    def plot_2D_image_comparison(self, raw_image, pred_image, cmap, cbar_label, title, savefig=None):

        plt.figure(figsize = (18,5))

        plt.subplot(1, 3, 1)
        ax1 = plt.gca()
        if cmap:
            plt.imshow(raw_image,cmap = cmap,vmin=0,vmax=np.max(raw_image))
        else:
            plt.imshow(raw_image,vmin=0,vmax=np.max(raw_image))
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
            plt.imshow(pred_image,cmap = cmap,vmin=0,vmax=np.max(pred_image))
        else:
            plt.imshow(pred_image,vmin=0,vmax=np.max(pred_image))
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.title('Prediction image', fontsize=14)
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        abs_diff = np.abs(raw_image-pred_image)
        plt.subplot(1, 3, 3)
        ax3 = plt.gca()
        if cmap:
            plt.imshow(abs_diff,cmap = cmap,vmin=0,vmax=np.max(abs_diff))
        else:
            plt.imshow(abs_diff,vmin=0,vmax=np.max(abs_diff))
        plt.xlabel('X coordinate', fontsize=14)
        plt.ylabel('Y coordinate', fontsize=14)
        plt.title('Absoluate difference', fontsize=14)
        ax3.axes.get_xaxis().set_ticks([])
        ax3.axes.get_yaxis().set_ticks([])
        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.suptitle(title, fontsize=18)
        if savefig:
            plt.savefig(savefig,dpi=300,bbox_inches='tight')
        plt.show()

        

    def plot_animation_InSAR_AI_surveillance(self, video, title):
        """
        Visualization with animation for InSAR, ground truth properties, and AI predicted properties.
        video is a 4D shape of (3, num_tt, num_xx, num_yy)
        e.g. (3, 19, 200, 200)
        dim0=3. 
        dim0=0 --> InSAR
        dim0=1 --> Ground truth
        dim0=2 --> AI predictions
        """

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots



        data1 = video[0,:,:,:]
        data2 = video[1,:,:,:]
        data3 = video[2,:,:,:]

        # Create a figure with subplots
        fig = make_subplots(rows=1, cols=3, horizontal_spacing = 0.12,
                            subplot_titles=("InSAR, mm", "Ground truth", "AI prediction"))

        # Initialize subplots (using the first frame data here)
        fig.add_trace(go.Heatmap(z=data1[0, :, :], 
                                #  zmin=0,
                                #  zmax=100,
                                zauto=False,
                                coloraxis="coloraxis1"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=data2[0, :, :],
                                #  zmin=np.min(data2),
                                #  zmax=np.max(data2),
                                coloraxis="coloraxis2"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=data3[0, :, :],
                                #  zmin=np.min(data3),
                                #  zmax=np.max(data3),
                                coloraxis="coloraxis2"), row=1, col=3)


        # Customize layout for colorbars
        fig.update_layout(coloraxis1=dict(colorscale='Greys', colorbar=dict(x=0.28, len=1)),
                        coloraxis2=dict(colorscale='Greys', colorbar=dict(x=0.65, len=1)),
                        title_text=title,  # Super title text
                        title_x=0.5,  # Align the super title to center
                        title_font=dict(size=24),  #
                        width=1200, height=445)



        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        fig.update_layout(
        {
            "coloraxis1_cmin": np.min(data1),
            "coloraxis1_cmax": np.max(data1)*0.9,
            "coloraxis2_cmin": np.min(data2),
            "coloraxis2_cmax": np.max(data2),
        }
                        )

        # Add frames for the animation
        frames = [go.Frame(data=[go.Heatmap(z=data1[i, :, :], coloraxis="coloraxis1"),
                                go.Heatmap(z=data2[i, :, :], coloraxis="coloraxis2"),
                                go.Heatmap(z=data3[i, :, :], coloraxis="coloraxis2")])
                for i in range(data1.shape[0])]

        fig.frames = frames

        for k in range(len(fig.frames)):
            fig.frames[k]['layout'].update(title_text=f'{title} after {k} years')

        # Add slider and play button for animation control
        fig.update_layout(updatemenus=[dict(type="buttons", showactive=False,
                                            y=0, x=0.5, xanchor="center", yanchor="top",
                                            buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None, dict(frame=dict(duration=500, redraw=True), 
                                                                        fromcurrent=True, mode="immediate")])])],
                            
                        #   sliders=[dict(steps=[dict(method="animate", args=[[f.name], 
                        #                                                      dict(mode="immediate", frame=dict(duration=500, redraw=True))]) 
                        #                        for f in fig.frames])]
                                            )

        fig.show()

