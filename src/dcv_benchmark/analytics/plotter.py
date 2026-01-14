from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dcv_benchmark.models.metrics import SecurityMetrics

ASR_PLOT_FILENAME = "asr_by_strategy.png"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
LATENCY_PLOT_FILENAME = "latency_distribution.png"
PLOT_DIR = "plots"


class Plotter:
    """
    Generates plots for an experiment.
    """

    def __init__(self, output_dir: Path):
        self.plots_dir = output_dir / PLOT_DIR
        self.plots_dir.mkdir(exist_ok=True, parents=True)

    def generate_all(self, metrics: SecurityMetrics) -> None:
        """Generates all standard plots for a security run."""
        self._plot_confusion_matrix(metrics)
        self._plot_strategy_asr(metrics)
        self._plot_latency_distribution(metrics)

    def _plot_confusion_matrix(self, metrics: SecurityMetrics) -> None:
        """
        Generates a 2x2 Confusion Matrix Heatmap.

        Axes:
          Y = Actual (Attack vs Benign)
          X = Predicted (Detected vs Safe)
        """
        gm = metrics.global_metrics
        matrix = np.array(
            [
                [gm.tp, gm.fn],  # Actual Attack
                [gm.fp, gm.tn],  # Actual Benign
            ]
        )

        labels = np.array(
            [[f"TP\n{gm.tp}", f"FN\n{gm.fn}"], [f"FP\n{gm.fp}", f"TN\n{gm.tn}"]]
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot heatmap
        im = ax.imshow(matrix, cmap="Blues", vmin=0)

        # Add labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Detected\n(Block)", "Safe\n(Allow)"])
        ax.set_yticklabels(["Attack\nSample", "Benign\nSample"])

        ax.set_xlabel("SDK Verdict")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Detection Confusion Matrix")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i, j], ha="center", va="center", color="black")

        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(self.plots_dir / CONFUSION_MATRIX_FILENAME)
        plt.close()

    def _plot_strategy_asr(self, metrics: SecurityMetrics) -> None:
        """
        Horizontal bar chart of Attack Success Rate (Failure Rate) by Strategy.
        """
        if not metrics.by_strategy:
            return

        strategies = list(metrics.by_strategy.keys())
        asr_scores = [m.asr for m in metrics.by_strategy.values()]

        # Sort by ASR (Most dangerous at top)
        sorted_indices = np.argsort(asr_scores)
        strategies = [strategies[i] for i in sorted_indices]
        asr_scores = [asr_scores[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(8, len(strategies) * 0.5 + 2))

        y_pos = np.arange(len(strategies))
        bars = ax.barh(y_pos, asr_scores, align="center", color="#d62728")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(strategies)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Attack Success Rate (ASR)")
        ax.set_title("Vulnerability by Attack Strategy")

        # Add value labels
        ax.bar_label(bars, fmt="%.2f", padding=3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / ASR_PLOT_FILENAME)
        plt.close()

    def _plot_latency_distribution(self, metrics: SecurityMetrics) -> None:
        """
        Comparing latency distributions of attack vs. benign samples.
        Useful to see if the defense adds overhead or if attacks cause timeouts.
        """
        benign = metrics.global_metrics.latencies_benign
        attack = metrics.global_metrics.latencies_attack

        if not benign and not attack:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot overlapping histograms
        ax.hist(benign, bins=20, alpha=0.5, label="Benign", color="blue", density=True)
        ax.hist(attack, bins=20, alpha=0.5, label="Attack", color="red", density=True)

        ax.set_xlabel("Latency (seconds)")
        ax.set_ylabel("Density")
        ax.set_title("Latency Distribution: Benign vs. Attack")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / LATENCY_PLOT_FILENAME)
        plt.close()
