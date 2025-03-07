import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, filename):
    """Erstellt eine Confusion Matrix und speichert sie als Bild."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()
