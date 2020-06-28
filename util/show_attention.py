import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def show_attention(ax, fig, attention, input_condition, sentence, if_show=False):

    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([" "] + input_condition, rotation=90)
    ax.set_yticklabels([" "] + sentence)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if if_show:
        plt.show()
    return ax