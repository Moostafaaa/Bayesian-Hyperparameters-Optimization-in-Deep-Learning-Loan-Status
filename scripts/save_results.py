"""
save_results.py
---------------
Run after study.optimize() completes to persist best parameters
and export Optuna visualization plots.

Usage:
    python scripts/save_results.py
    (assumes `study` is already defined in the same Python session,
     or load from an Optuna RDB storage if you used one)
"""

import json
import os
import optuna
import optuna.visualization as vis


def save_study_results(study: optuna.Study, output_dir: str = "results") -> None:
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # Save best hyperparameters as JSON
    params_path = f"{output_dir}/best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"Best params saved → {params_path}")
    print(f"Best AUC: {study.best_value:.6f}")

    # Export Optuna plots (requires kaleido: pip install kaleido)
    try:
        vis.plot_optimization_history(study).write_image(
            f"{output_dir}/plots/optimization_history.png"
        )
        vis.plot_param_importances(study).write_image(
            f"{output_dir}/plots/param_importances.png"
        )
        vis.plot_parallel_coordinate(study).write_image(
            f"{output_dir}/plots/parallel_coordinates.png"
        )
        print(f"Plots saved → {output_dir}/plots/")
    except Exception as e:
        print(f"Plot export failed (is kaleido installed?): {e}")


if __name__ == "__main__":
    # Example: load from an existing RDB study
    # study = optuna.load_study(study_name="...", storage="sqlite:///optuna_study.db")
    print("Import save_study_results and call it with your study object.")
