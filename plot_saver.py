import matplotlib.pyplot as plt

def save_loss_plot(actor_losses, critic_losses, filename="loss_plot.png"):
    """
    Saves a Matplotlib plot with two subplots: one for actor loss, one for critic loss.

    Parameters:
    -----------
    actor_losses : list or array-like
        The actor loss values at each update step.
    critic_losses : list or array-like
        The critic loss values at each update step.
    filename : str
        The name of the file to save the figure as (e.g. 'loss_plot.png').
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(actor_losses, label="Actor Loss", color='blue')
    ax1.set_ylabel("Loss")
    ax1.set_title("Actor Loss")
    ax1.legend()

    ax2.plot(critic_losses, label="Critic Loss", color='orange')
    ax2.set_xlabel("Update Steps")
    ax2.set_ylabel("Loss")
    ax2.set_title("Critic Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
