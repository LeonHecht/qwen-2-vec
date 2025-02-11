import torch
import matplotlib.pyplot as plt


def plot_training_metrics():

    # Load saved log history from the HF Trainer.
    # This file should contain a list of dictionaries with logged metrics.
    log_history = torch.load("training_metrics_hf.pth")

    # Initialize lists to store metrics
    train_losses = []
    eval_losses = []
    grad_norms = []

    # Iterate over each log entry and extract the metrics.
    for entry in log_history:
        # Check if a training loss is logged.
        if "loss" in entry:
            train_losses.append(entry["loss"])
        # Check if an evaluation loss is logged.
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
        # Check if a gradient norm is logged.
        if "grad_norm" in entry:
            grad_norms.append(entry["grad_norm"])

    # Plot training losses and evaluation losses in one plot.
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Eval Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.show()

    # Plot gradient norms in a separate plot.
    plt.figure(figsize=(10, 5))
    plt.plot(grad_norms, label="Gradient Norms")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_training_metrics()