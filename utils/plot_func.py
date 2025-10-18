import matplotlib.pyplot as plt
import matplotlib.axes as axes
import math
import numpy as np

def plot_1D(x, y, plottype, xlabel=None, ylabel=None, 
            extparams=str):

    fig = plt.figure(figsize=(6,4))

    if plottype == "scatter":
        plt.scatter(x, y, s=10, color="blue")
    elif plottype == "line":
        plt.plot(x, y, linewidth=2, color="blue")

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    titlestr = ylabel + " against " + xlabel + extparams
    plt.title(titlestr, fontsize=16)
    plt.grid()

    return fig


def plot_collate(x, y, plottype, xlabel=None, ylabel=None, handles=None):

    fig = plt.figure(figsize=(6,4))
    size = len(y[0,:])

    if plottype == "scatter":
        for i in range(size):
            plt.scatter(x[:,i], y[:,i], s=5, label=handles[i])

    elif plottype == "line":
        for i in range(size):
            plt.plot(x[:,i], y[:,i], linewidth=1.5, label=handles[i])
    
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid()
    plt.legend(handles, loc="upper right", fontsize=12)

    return fig


def plot_compare(x, y, xlabel, ylabel, root_index=0):

    fig = plt.figure(figsize=(6,4))

    plt.scatter(x, y[:,root_index], s=3, color="red", label="root") # plot the root data
    if root_index == 0:
        plt.plot(x, y[:,1], linewidth=2, label="estimated")         # plot all other data    
    else:
        plt.plot(x, y[:,0], linewidth=2, label="estimated")         # plot all other data    

    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.grid()

    return fig


def plot_compare_multiple(x, y_root, y_estimate, indexes, 
                          xlabel, ylabel, 
                          root_index=0, extparams=""):
    k = len(indexes)

    # determine dimension of subplot
    if k <= 3:
        fig, axs = plt.subplots(k)
        rows = 1; cols = k
    elif k == 4:
        fig, axs = plt.subplots(2, 2)
        rows = 2; cols = 2
    else:
        rows = math.ceil(k/3); cols = 3
        fig, axs = plt.subplots(int(rows), 3)

    rows = int(rows); cols = int(cols)
    titlestr = ylabel + " against " + xlabel + extparams
    fig.suptitle(titlestr, fontsize=14)

    # plot subplots
    count = 0
    for i in range(rows):
        for j in range(cols):
            index = indexes[count]

            if k <= 3:
                ax = axs[i+j]
            else:
                ax = axs[i, j]

            # subplot
            ax.scatter(x, y_root[:, index], s=5, color="blue", label="root")
            ax.plot(x, y_estimate[:, index], linewidth=1.5, color="red", label="estimate")
            ax.set_title(f"Mode {count+1}", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid()

            count += 1
            if count >= len(indexes):
                break
    
    plt.tight_layout()

    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)

    return fig, axs


def plot_score(scores, plottype, xlabel):

    x = np.linspace(1, len(scores), num=len(scores))
    new_list = range(1, len(scores))

    fig = plt.figure(figsize=(6,4))

    if plottype == "scatter":
        plt.scatter(x, scores, s=7, color="blue")
    elif plottype == "line":
        plt.plot(x, scores, linewidth=1.5, color="blue")

    plt.plot()

    plt.xticks(new_list)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("R2 score", fontsize=16)
    plt.grid()

    return fig


def plot_compare_multiple_xdotx(x, y_root, y_estimate, indexes, xlabel, ylabel, root_index=0, extparams=""):
    k = len(indexes)

    # determine dimension of subplot
    if k <= 3:
        fig, axs = plt.subplots(k)
        rows = 1; cols = k
    elif k == 4:
        fig, axs = plt.subplots(2, 2)
        rows = 2; cols = 2
    else:
        rows = math.ceil(k/3); cols = 3
        fig, axs = plt.subplots(int(rows), 3)

    rows = int(rows); cols = int(cols)

    titlestr = ylabel + " against " + xlabel + extparams
    fig.suptitle(titlestr, fontsize=14)

    # plot subplots
    count = 0
    for i in range(rows):
        for j in range(cols):
            index = indexes[count]

            if k <= 3:
                ax = axs[i+j]
            else:
                ax = axs[i, j]

            # subplot
            ax.scatter(x[:, index], y_root[:, index], s=5, color="blue", label="root")
            ax.plot(x[:, index], y_estimate[:, index], linewidth=1.5, color="red", label="estimate")
            ax.set_title(f"Mode {i+j+1}", fontsize=12)
            ax.legend(fontsize=12)
            ax.grid()

            count += 1
            if count >= len(indexes):
                break
    
    plt.tight_layout()
    
    for ax in axs.flat:
        ax.set(xlabel=xlabel, ylabel=ylabel)
    
    return fig, axs

