
"""
quick_run.py  –  fast sanity-check run with tiny hyperparams
Useful for verifying the pipeline before a full overnight run.
"""
import subprocess, sys

cmd = [
    sys.executable, "scripts/autoreg_lim.py",
    "--task", "task1",
    "--n_values", "2", "4", "8",
    "--n_layers_values", "1", "2",
    "--n_heads", "2",
    "--d_model", "64",
    "--d_ff", "256",
    "--train_samples", "10000",
    "--eval_samples", "1000",
    "--epochs", "5",
    "--batch_size", "32",
    "--output_dir", "./results_quick",
    "--modes", "both",
]
print("Running quick test:", " ".join(cmd))
subprocess.run(cmd, check=True)