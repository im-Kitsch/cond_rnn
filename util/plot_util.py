#%%

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

import matplotlib.pyplot as plt


def plot_arrow_2d(ax, traj, ground_truth=None):
    x1, y1 = traj[:, 0], traj[:, 1]
    ax.quiver(x1[:-1], y1[:-1], x1[1:] - x1[:-1], y1[1:] - y1[:-1], color="red",
              scale_units='xy', angles='xy', scale=1)
    ax.scatter(x1[0], y1[0], marker="*", s=380)
    if not (ground_truth is None):
        x2, y2 = ground_truth[:, 0], ground_truth[:, 1]
        ax.quiver(x2[:-1], y2[:-1], x2[1:] - x2[:-1], y2[1:] - y2[:-1], color="black",
                  scale_units='xy', angles='xy', scale=1)
        ax.scatter(x2[0], y2[0], marker="*", s=380)
    return ax


def plot_arrow_3d(ax, traj, ground_truth=None):
    x1, y1 = traj[:, 0], traj[:, 1]
    mpl.rcParams['legend.fontsize'] = 10

    t = np.arange(x1.shape[0])
    ax.plot(x1, y1, t, label='trajectory')
    for i in range(x1.shape[0] - 1):
        ax.plot([x1[i], x1[i + 1]], [y1[i], y1[i + 1]], [i, i + 1])
    ax.scatter(x1, y1, t)

    if not (ground_truth is None):
        x2, y2 = ground_truth[:, 0], ground_truth[:, 1]
        t = np.arange(x2.shape[0])
        ax.plot(x2, y2, t, label='ground_truth')
        ax.scatter(x2, y2, t)
        for i in range(x2.shape[0] - 1):
            ax.plot([x2[i], x2[i + 1]], [y2[i], y2[i + 1]], [i, i + 1])

    # ax.legend()
    return ax


def visualization_trajectory(trajectories, n_row, n_col, plot_3d=True):
    assert trajectories.ndim == 3
    n_traj = trajectories.shape[0]
    assert n_row * n_col >= n_traj

    fig1, axes1 = plt.subplots(nrows=n_row, ncols=n_col)
    if n_row * n_col != 1:
        axes1 = axes1.flatten()
    else:
        axes1 = np.array(axes1).flatten()
    for _i in range(n_traj):
        ax = axes1[_i]
        traj = trajectories[_i]
        plot_arrow_2d(ax, traj)

    if plot_3d:
        fig2 = plt.figure()
        for _i in range(n_traj):
            ax = fig2.add_subplot(n_row, n_col, _i+1, projection="3d")
            traj = trajectories[_i]
            plot_arrow_3d(ax, traj)
    else:
        fig2 = None
    return fig1, fig2


if __name__ == "__main__":
    fig1, fig2 = visualization_trajectory(np.random.random((1, 8, 2)), 1, 1)
    plt.show()

