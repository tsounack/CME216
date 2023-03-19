# Import libraries
import numpy as np
import wandb
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable


def l2_pt_error(pred, true):
    return (np.linalg.norm(pred - true) / np.linalg.norm(true))*100

def plot_hist(rel_error, epoch):
    """
    Plot a histogram of the relative error of individual samples in the validation
    set
    """
    fig = px.histogram(rel_error, range_x=[0,100])
    fig.update_layout(
        xaxis_title = "Relative error in %",
        yaxis_title = "Counts"
    )
    wandb.log({"rel_error_hist": fig, "epoch": epoch})



def plot_perm_and_temp(m_field, u_field, model_pred, epoch):
    """
    Plot the pressure (u_field) and the permeability (m_field)
    """
    cmap = "inferno"
    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(5):
        im = axs[0, i].imshow(m_field[i, :, :], cmap=cmap)
        divider = make_axes_locatable(axs[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=axs[0, i], cax=cax)
        axs[0, i].set_title(f"Perm. {i}")
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        axs[1, i].imshow(u_field[i, :, :], cmap=cmap)
        divider = make_axes_locatable(axs[1, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=axs[1, i], cax=cax)
        axs[1, i].set_title(f"True pressure. {i}")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        axs[2, i].imshow(model_pred[i, :, :].squeeze(), cmap=cmap)
        divider = make_axes_locatable(axs[2, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=axs[2, i], cax=cax)
        axs[2, i].set_title(f"Pred. pressure. {i}")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

        diff = model_pred - u_field
        axs[3, i].imshow(diff[i, :, :], cmap=cmap)
        divider = make_axes_locatable(axs[3, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=axs[3, i], cax=cax)
        axs[3, i].set_title(f"Error = {l2_pt_error(model_pred[i, :, :], u_field[i, :, :]):.2f}%")
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])

    plt.tight_layout()
    wandb.log({"true_pred_pressure": fig, "epoch": epoch})
