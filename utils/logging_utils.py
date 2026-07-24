"""
Experiment logging utilities.

Supports CSV logging and optional wandb integration.
"""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Logger that writes results to CSV and optionally to wandb."""

    def __init__(
        self,
        results_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
        wandb_project: str = 'trace',
    ):
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        os.makedirs(results_dir, exist_ok=True)

        self.csv_path = os.path.join(results_dir, f"{experiment_name}.csv")
        self.log_path = os.path.join(results_dir, f"{experiment_name}.log")
        self._csv_writer = None
        self._csv_file = None
        self._fieldnames = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=experiment_name)
                self._wandb = wandb
            except ImportError:
                print("wandb not installed, falling back to CSV only")
                self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dictionary of metrics."""
        # CSV
        if self._csv_writer is None:
            self._fieldnames = ['timestamp', 'step'] + sorted(metrics.keys())
            self._csv_file = open(self.csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()

        row = {'timestamp': datetime.now().isoformat(), 'step': step}
        row.update(metrics)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # wandb
        if self.use_wandb:
            self._wandb.log(metrics, step=step)

    def log_text(self, message: str):
        """Log a text message."""
        timestamp = datetime.now().isoformat()
        with open(self.log_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

    def save_results(self, results: Dict[str, Any], filename: str = 'results.json'):
        """Save final results as JSON."""
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def close(self):
        """Close all file handles."""
        if self._csv_file:
            self._csv_file.close()
        if self.use_wandb:
            self._wandb.finish()


def format_results_table(
    results: Dict,
    datasets: List[str],
    models: List[str],
    explainers: List[str],
    metrics: List[str],
) -> str:
    """Format results as a readable text table."""
    lines = []
    header = f"{'Dataset':<12} {'Model':<12} {'Explainer':<12} " + " ".join(f"{m:>12}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    for ds in datasets:
        for model in models:
            for exp in explainers:
                key = f"{ds}_{model}_{exp}"
                if key in results:
                    r = results[key]
                    vals = " ".join(f"{r.get(m, 'N/A'):>12.4f}" if isinstance(r.get(m), float) else f"{str(r.get(m, 'N/A')):>12}" for m in metrics)
                    lines.append(f"{ds:<12} {model:<12} {exp:<12} {vals}")

    return "\n".join(lines)
