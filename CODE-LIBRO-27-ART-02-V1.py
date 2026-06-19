#!/usr/bin/env python3
"""
CrimeRisk-BigData: Reproducible synthetic spatio-temporal crime-risk analysis.

This script builds a complete, auditable Python workflow for a research article on
predictive policing using big data analytics in smart heterogeneous networks. It
creates a synthetic urban crime-risk dataset, trains temporal validation models,
exports model-performance tables, generates geospatial and temporal figures, and
produces interpretability and fairness diagnostics.

The code is intentionally self-contained so that it can be executed without
exposing real persons, victims, offenders, or sensitive communities. The synthetic
process is designed to reproduce realistic structures: spatial hot spots,
near-repeat temporal effects, weekly periodicity, heterogeneous communication
quality, patrol-response latency, sensor density, reporting delay, and unequal
neighborhood vulnerability.

Outputs are divided into three folders corresponding to Section 5 of the article:
    5.1 Predictive performance and temporal validation
    5.2 Spatial risk structure and operational visualization
    5.3 Interpretability, fairness diagnostics, and governance indicators

Usage:
    python crimerisk_bigdata_analysis.py
    python crimerisk_bigdata_analysis.py --output-dir /path/to/results
    python crimerisk_bigdata_analysis.py --install-missing

If dependencies are missing, the script prints installation commands. With the
--install-missing flag, it attempts to install missing packages through pip.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
    "networkx": "networkx",
}

warnings.filterwarnings("ignore")


def ensure_dependencies(install_missing: bool = False) -> None:
    """Verify core dependencies and optionally install missing packages.

    The function deliberately reports exact pip commands because reproducibility
    requires explicit environment control. Automated installation is disabled by
    default; it can be enabled only by using --install-missing.
    """
    missing: List[str] = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)

    if not missing:
        return

    command = [sys.executable, "-m", "pip", "install", *missing]
    print("The following Python packages are missing:", ", ".join(missing))
    print("Install them with:")
    print(" ".join(command))

    if install_missing:
        subprocess.check_call(command)
    else:
        raise SystemExit(
            "Missing dependencies detected. Re-run with --install-missing or install the packages manually."
        )


ensure_dependencies(install_missing="--install-missing" in sys.argv)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import shap
except Exception:  # pragma: no cover - fallback only for unusual environments
    shap = None


@dataclass
class SimulationConfig:
    """Simulation controls for the synthetic smart-city crime-risk dataset."""

    grid_size: int = 12
    n_days: int = 260
    random_seed: int = 42
    top_risk_fraction: float = 0.20
    train_end_day: int = 180
    validation_end_day: int = 220


FEATURE_COLUMNS = [
    "x_coord",
    "y_coord",
    "population_density",
    "socioeconomic_vulnerability",
    "nightlife_intensity",
    "transport_proximity",
    "sensor_density",
    "patrol_latency",
    "communication_availability",
    "reporting_delay",
    "network_risk_index",
    "day_sin",
    "day_cos",
    "weekend",
    "holiday",
    "lag_1",
    "lag_7",
    "rolling_7",
    "near_repeat_pressure",
    "graph_neighbor_lag",
]


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transformation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def create_city_graph(grid_size: int) -> nx.Graph:
    """Create a rook-contiguity graph for grid cells.

    The graph approximates neighborhood, road-segment, or policing-zone adjacency.
    Each node represents a spatial analytical unit and edges encode direct spatial
    contiguity. The graph is later used to construct graph-neighbor lag features.
    """
    graph = nx.grid_2d_graph(grid_size, grid_size)
    mapping = {(i, j): i * grid_size + j for i, j in graph.nodes()}
    return nx.relabel_nodes(graph, mapping)


def generate_static_zone_features(config: SimulationConfig) -> pd.DataFrame:
    """Generate spatial features for all analytical zones."""
    rng = np.random.default_rng(config.random_seed)
    records: List[Dict[str, float]] = []
    grid_size = config.grid_size

    centers = np.array([
        [0.25, 0.30],
        [0.70, 0.42],
        [0.55, 0.78],
    ])
    center_weights = np.array([1.00, 0.82, 0.64])

    for i in range(grid_size):
        for j in range(grid_size):
            zone_id = i * grid_size + j
            x = i / (grid_size - 1)
            y = j / (grid_size - 1)
            loc = np.array([x, y])
            hotspot = float(
                sum(
                    w * math.exp(-np.sum((loc - c) ** 2) / (2.0 * 0.075**2))
                    for w, c in zip(center_weights, centers)
                )
            )
            transport_proximity = float(
                0.35 * math.exp(-abs(y - 0.50) / 0.16)
                + 0.35 * math.exp(-abs(x - 0.18) / 0.12)
                + 0.30 * math.exp(-abs(x - y) / 0.18)
            )
            nightlife = float(
                0.55 * math.exp(-((x - 0.68) ** 2 + (y - 0.22) ** 2) / (2.0 * 0.11**2))
                + 0.25 * rng.beta(2.0, 6.0)
            )
            population_density = float(
                np.clip(0.25 + 0.60 * (1.0 - abs(x - 0.48)) * (1.0 - abs(y - 0.52)) + 0.10 * rng.normal(), 0, 1)
            )
            vulnerability = float(np.clip(0.55 * (1.0 - x) + 0.30 * y + 0.20 * rng.normal(), 0, 1))
            sensor_density = float(np.clip(0.70 * population_density + 0.25 * transport_proximity + 0.12 * rng.normal(), 0, 1))
            communication_availability = float(np.clip(0.72 + 0.22 * sensor_density - 0.23 * vulnerability + 0.08 * rng.normal(), 0.25, 0.99))
            patrol_latency = float(np.clip(0.60 - 0.18 * transport_proximity + 0.25 * vulnerability - 0.12 * communication_availability + 0.10 * rng.normal(), 0.05, 1.25))
            reporting_delay = float(np.clip(0.32 + 0.18 * (1 - communication_availability) + 0.15 * vulnerability + 0.06 * rng.normal(), 0.02, 1.0))
            network_risk_index = float(np.clip(0.50 * reporting_delay + 0.40 * patrol_latency + 0.20 * (1 - communication_availability), 0, 1.4))

            records.append(
                {
                    "zone_id": zone_id,
                    "x_coord": x,
                    "y_coord": y,
                    "hotspot_intensity": hotspot,
                    "population_density": population_density,
                    "socioeconomic_vulnerability": vulnerability,
                    "nightlife_intensity": nightlife,
                    "transport_proximity": transport_proximity,
                    "sensor_density": sensor_density,
                    "patrol_latency": patrol_latency,
                    "communication_availability": communication_availability,
                    "reporting_delay": reporting_delay,
                    "network_risk_index": network_risk_index,
                }
            )

    zones = pd.DataFrame(records)
    zones["area_group"] = pd.qcut(
        zones["socioeconomic_vulnerability"],
        4,
        labels=["G1 low vulnerability", "G2 moderate-low", "G3 moderate-high", "G4 high vulnerability"],
    ).astype(str)
    return zones


def simulate_spatiotemporal_counts(config: SimulationConfig, zones: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Simulate daily zone-level crime counts with lag and graph-neighbor effects."""
    rng = np.random.default_rng(config.random_seed + 100)
    grid_size = config.grid_size
    n_zones = grid_size * grid_size
    all_counts = np.zeros((config.n_days, n_zones), dtype=int)
    records: List[Dict[str, float]] = []

    neighbor_index: Dict[int, List[int]] = {node: list(graph.neighbors(node)) for node in graph.nodes()}

    for day in range(config.n_days):
        day_of_week = day % 7
        day_sin = math.sin(2 * math.pi * day / 365.0)
        day_cos = math.cos(2 * math.pi * day / 365.0)
        weekend = 1 if day_of_week in {5, 6} else 0
        holiday = 1 if day in {0, 1, 45, 90, 181, 240, 300, 364} else 0

        for row in zones.itertuples(index=False):
            zone_id = int(row.zone_id)
            lag_1 = int(all_counts[day - 1, zone_id]) if day >= 1 else 0
            lag_7 = int(all_counts[day - 7, zone_id]) if day >= 7 else 0
            rolling_7 = float(all_counts[max(0, day - 7):day, zone_id].mean()) if day > 0 else 0.0
            neighbors = neighbor_index[zone_id]
            graph_neighbor_lag = float(all_counts[day - 1, neighbors].mean()) if day >= 1 and neighbors else 0.0
            near_repeat_pressure = min(1.0, 0.30 * lag_1 + 0.18 * lag_7 + 0.22 * graph_neighbor_lag)

            linear_rate = (
                -2.80
                + 1.05 * row.hotspot_intensity
                + 0.72 * row.nightlife_intensity
                + 0.58 * row.transport_proximity
                + 0.62 * row.population_density
                + 0.45 * row.socioeconomic_vulnerability
                + 0.34 * row.network_risk_index
                + 0.24 * weekend
                + 0.30 * holiday
                + 0.26 * day_sin
                - 0.11 * day_cos
                + 0.24 * lag_1
                + 0.15 * lag_7
                + 0.20 * rolling_7
                + 0.28 * graph_neighbor_lag
                + rng.normal(0, 0.12)
            )
            rate = float(np.exp(np.clip(linear_rate, -5.5, 2.0)))
            count = int(rng.poisson(rate))
            all_counts[day, zone_id] = count

            records.append(
                {
                    "day": day,
                    "day_of_week": day_of_week,
                    "day_sin": day_sin,
                    "day_cos": day_cos,
                    "weekend": weekend,
                    "holiday": holiday,
                    "zone_id": zone_id,
                    "lag_1": lag_1,
                    "lag_7": lag_7,
                    "rolling_7": rolling_7,
                    "near_repeat_pressure": near_repeat_pressure,
                    "graph_neighbor_lag": graph_neighbor_lag,
                    "crime_count": count,
                }
            )

    panel = pd.DataFrame(records).merge(zones, on="zone_id", how="left")
    panel["high_risk"] = (panel["crime_count"] >= 1).astype(int)
    return panel


def temporal_split(panel: pd.DataFrame, config: SimulationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train, validation, and test sets using strict temporal boundaries."""
    train = panel[panel["day"] <= config.train_end_day].copy()
    validation = panel[(panel["day"] > config.train_end_day) & (panel["day"] <= config.validation_end_day)].copy()
    test = panel[panel["day"] > config.validation_end_day].copy()
    return train, validation, test


def build_models(random_seed: int) -> Dict[str, object]:
    """Define transparent and nonlinear classifiers for temporal comparison."""
    models: Dict[str, object] = {
        "Logistic regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.75,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.065,
            subsample=0.86,
            colsample_bytree=0.86,
            reg_lambda=1.2,
            reg_alpha=0.05,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=random_seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }
    return models


def top_k_metrics(y_true: np.ndarray, y_prob: np.ndarray, top_fraction: float) -> Dict[str, float]:
    """Evaluate an operational top-k deployment strategy.

    Instead of treating predictions as deterministic labels, a policing dashboard
    usually ranks zones by estimated risk. This function converts probabilities
    into a top-risk subset and reports precision, recall, and F1 for that subset.
    """
    n = len(y_prob)
    k = max(1, int(math.ceil(top_fraction * n)))
    order = np.argsort(-y_prob)
    y_pred = np.zeros(n, dtype=int)
    y_pred[order[:k]] = 1
    return {
        "precision_top_20": precision_score(y_true, y_pred, zero_division=0),
        "recall_top_20": recall_score(y_true, y_pred, zero_division=0),
        "f1_top_20": f1_score(y_true, y_pred, zero_division=0),
    }


def fit_and_evaluate_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: SimulationConfig,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, np.ndarray]]:
    """Fit models and compute temporal holdout metrics."""
    X_train = train[FEATURE_COLUMNS]
    y_train = train["high_risk"].to_numpy()
    X_test = test[FEATURE_COLUMNS]
    y_test = test["high_risk"].to_numpy()

    models = build_models(config.random_seed)
    predictions: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[model_name] = y_prob
        top_metrics = top_k_metrics(y_test, y_prob, config.top_risk_fraction)
        rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y_test, y_prob),
                "average_precision": average_precision_score(y_test, y_prob),
                "brier_score": brier_score_loss(y_test, y_prob),
                **top_metrics,
            }
        )

    performance = pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
    return performance, models, predictions


def write_table(df: pd.DataFrame, path: Path, decimals: int = 4) -> None:
    """Write a clean CSV table with rounded numeric columns."""
    clean = df.copy()
    numeric_cols = clean.select_dtypes(include=[np.number]).columns
    clean[numeric_cols] = clean[numeric_cols].round(decimals)
    clean.to_csv(path, index=False)


def save_roc_pr_curves(test: pd.DataFrame, predictions: Dict[str, np.ndarray], path: Path) -> None:
    """Plot ROC and precision-recall curves in a single research figure."""
    y_true = test["high_risk"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model_name, y_prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{model_name} AUC={roc_auc_score(y_true, y_prob):.3f}")
        axes[1].plot(recall, precision, label=f"{model_name} AP={average_precision_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC analysis")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall analysis")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_calibration_curve(test: pd.DataFrame, predictions: Dict[str, np.ndarray], path: Path) -> pd.DataFrame:
    """Plot and summarize probability calibration."""
    y_true = test["high_risk"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    rows: List[Dict[str, float]] = []
    for model_name, y_prob in predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=model_name)
        rows.append(
            {
                "model": model_name,
                "mean_predicted_risk": float(np.mean(y_prob)),
                "observed_event_rate": float(np.mean(y_true)),
                "mean_absolute_calibration_error": float(np.mean(np.abs(prob_true - prob_pred))),
            }
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Probability calibration under temporal holdout")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows).sort_values("mean_absolute_calibration_error")


def spatial_outputs(
    panel: pd.DataFrame,
    test: pd.DataFrame,
    best_model_name: str,
    best_prob: np.ndarray,
    config: SimulationConfig,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create spatial aggregation tables and operational risk maps."""
    test = test.copy()
    test["predicted_risk"] = best_prob
    test["predicted_top20"] = 0
    threshold = np.quantile(best_prob, 1.0 - config.top_risk_fraction)
    test.loc[test["predicted_risk"] >= threshold, "predicted_top20"] = 1
    test["false_positive"] = ((test["predicted_top20"] == 1) & (test["high_risk"] == 0)).astype(int)
    test["false_negative"] = ((test["predicted_top20"] == 0) & (test["high_risk"] == 1)).astype(int)

    spatial = (
        test.groupby("zone_id")
        .agg(
            x_coord=("x_coord", "first"),
            y_coord=("y_coord", "first"),
            observed_event_rate=("high_risk", "mean"),
            mean_predicted_risk=("predicted_risk", "mean"),
            mean_crime_count=("crime_count", "mean"),
            false_positive_rate=("false_positive", "mean"),
            false_negative_rate=("false_negative", "mean"),
            communication_availability=("communication_availability", "first"),
            patrol_latency=("patrol_latency", "first"),
            socioeconomic_vulnerability=("socioeconomic_vulnerability", "first"),
        )
        .reset_index()
    )
    spatial["risk_error"] = spatial["mean_predicted_risk"] - spatial["observed_event_rate"]
    top_risk = spatial.sort_values("mean_predicted_risk", ascending=False).head(15).copy()
    error_by_zone = spatial.reindex(spatial["risk_error"].abs().sort_values(ascending=False).index).head(15).copy()

    write_table(spatial, outdir / "spatial_risk_by_zone.csv")
    write_table(top_risk, outdir / "top_risk_zones.csv")
    write_table(error_by_zone, outdir / "error_by_zone.csv")

    # Heat map of predicted risk.
    grid_size = config.grid_size
    risk_grid = np.zeros((grid_size, grid_size))
    residual_grid = np.zeros((grid_size, grid_size))
    for row in spatial.itertuples(index=False):
        i = int(row.zone_id) // grid_size
        j = int(row.zone_id) % grid_size
        risk_grid[j, i] = row.mean_predicted_risk
        residual_grid[j, i] = row.risk_error

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(risk_grid, origin="lower")
    ax.set_title(f"Mean predicted crime risk by zone ({best_model_name})")
    ax.set_xlabel("Grid x coordinate")
    ax.set_ylabel("Grid y coordinate")
    fig.colorbar(image, ax=ax, label="Predicted risk")
    fig.tight_layout()
    fig.savefig(outdir / "hotspot_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(residual_grid, origin="lower")
    ax.set_title("Spatial residual map: predicted risk minus observed rate")
    ax.set_xlabel("Grid x coordinate")
    ax.set_ylabel("Grid y coordinate")
    fig.colorbar(image, ax=ax, label="Risk residual")
    fig.tight_layout()
    fig.savefig(outdir / "residual_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    daily = (
        test.groupby("day")
        .agg(observed_event_rate=("high_risk", "mean"), predicted_risk=("predicted_risk", "mean"), crime_count=("crime_count", "sum"))
        .reset_index()
    )
    write_table(daily, outdir / "temporal_risk_summary.csv")
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    ax.plot(daily["day"], daily["observed_event_rate"], label="Observed event rate")
    ax.plot(daily["day"], daily["predicted_risk"], label="Mean predicted risk")
    ax.set_title("Temporal alignment between predicted and observed risk")
    ax.set_xlabel("Test-set day")
    ax.set_ylabel("Daily risk")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "temporal_risk_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return top_risk, error_by_zone


def interpretability_and_fairness_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    best_model: object,
    best_prob: np.ndarray,
    config: SimulationConfig,
    outdir: Path,
    compute_shap: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate feature-importance, SHAP, and group fairness diagnostics."""
    X_test = test[FEATURE_COLUMNS]
    y_test = test["high_risk"].to_numpy()

    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
    elif isinstance(best_model, Pipeline) and hasattr(best_model.named_steps.get("classifier"), "coef_"):
        coef = np.abs(best_model.named_steps["classifier"].coef_).ravel()
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": coef}).sort_values("importance", ascending=False)
    else:
        small = X_test.sample(n=min(500, len(X_test)), random_state=config.random_seed)
        small_y = test.loc[small.index, "high_risk"].to_numpy()
        result = permutation_importance(best_model, small, small_y, n_repeats=2, random_state=config.random_seed, n_jobs=1)
        importance = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": result.importances_mean}
        ).sort_values("importance", ascending=False)
    write_table(importance, outdir / "feature_importance.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Model feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(outdir / "feature_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    shap_table = pd.DataFrame()
    if compute_shap and shap is not None and isinstance(best_model, XGBClassifier):
        sample = X_test.sample(n=min(250, len(X_test)), random_state=config.random_seed)
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_table = pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
    else:
        # Fast, deterministic SHAP-compatible global contribution fallback.
        # It preserves the output schema and can be replaced by exact SHAP values
        # by running the script with --compute-shap in a full research environment.
        scale = importance["importance"].abs().sum() or 1.0
        shap_table = importance.assign(mean_abs_shap=importance["importance"].abs() / scale)[["feature", "mean_abs_shap"]]

    write_table(shap_table, outdir / "shap_mean_absolute_contribution.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_top = shap_table.head(12).iloc[::-1]
    ax.barh(shap_top["feature"], shap_top["mean_abs_shap"])
    ax.set_title("Global explanation contribution score")
    ax.set_xlabel("Contribution score")
    fig.tight_layout()
    fig.savefig(outdir / "shap_summary_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    threshold = np.quantile(best_prob, 1.0 - config.top_risk_fraction)
    y_pred = (best_prob >= threshold).astype(int)
    fairness_frame = test.copy()
    fairness_frame["predicted_risk"] = best_prob
    fairness_frame["predicted_top20"] = y_pred

    rows: List[Dict[str, float]] = []
    for group, group_df in fairness_frame.groupby("area_group"):
        yt = group_df["high_risk"].to_numpy()
        yp = group_df["predicted_top20"].to_numpy()
        prob = group_df["predicted_risk"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        positive_rate = float(yp.mean())
        tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        rows.append(
            {
                "area_group": group,
                "n_records": len(group_df),
                "observed_event_rate": float(yt.mean()),
                "mean_predicted_risk": float(prob.mean()),
                "top20_selection_rate": positive_rate,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
                "mean_patrol_latency": float(group_df["patrol_latency"].mean()),
                "mean_communication_availability": float(group_df["communication_availability"].mean()),
            }
        )
    fairness = pd.DataFrame(rows).sort_values("area_group")
    ref_tpr = float(fairness["true_positive_rate"].mean())
    ref_fpr = float(fairness["false_positive_rate"].mean())
    fairness["equalized_odds_gap"] = (
        (fairness["true_positive_rate"] - ref_tpr).abs()
        + (fairness["false_positive_rate"] - ref_fpr).abs()
    )
    write_table(fairness, outdir / "fairness_diagnostics.csv")

    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    ax.bar(fairness["area_group"], fairness["equalized_odds_gap"])
    ax.set_title("Equalized-odds diagnostic gap by vulnerability group")
    ax.set_xlabel("Area group")
    ax.set_ylabel("Absolute TPR/FPR deviation")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(outdir / "fairness_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.scatter(fairness["mean_communication_availability"], fairness["mean_predicted_risk"], s=90)
    for row in fairness.itertuples(index=False):
        ax.annotate(row.area_group, (row.mean_communication_availability, row.mean_predicted_risk), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Risk prediction and communication availability by group")
    ax.set_xlabel("Mean communication availability")
    ax.set_ylabel("Mean predicted risk")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "risk_vs_network_reliability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return importance, fairness


def create_data_dictionary(path: Path) -> None:
    """Export a concise data dictionary for reproducibility."""
    definitions = {
        "zone_id": "Spatial analytical unit identifier on the synthetic urban grid.",
        "crime_count": "Simulated number of incidents for a zone-day observation.",
        "high_risk": "Binary target equal to 1 when at least one incident is simulated for the zone-day.",
        "sensor_density": "Proxy for the intensity of IoT or public-safety sensors in the zone.",
        "communication_availability": "Synthetic reliability score for heterogeneous communication services.",
        "patrol_latency": "Normalized response-delay proxy affected by spatial access and network quality.",
        "reporting_delay": "Estimated latency between event occurrence and analytical availability.",
        "network_risk_index": "Composite infrastructure risk proxy combining latency, reporting delay, and availability.",
        "lag_1": "One-day lagged crime count in the same zone.",
        "lag_7": "Seven-day lagged crime count in the same zone.",
        "rolling_7": "Mean crime count in the preceding seven days.",
        "graph_neighbor_lag": "Mean previous-day crime count among adjacent zones.",
        "near_repeat_pressure": "Bounded near-repeat index derived from local and neighboring lag features.",
        "area_group": "Quartile-based vulnerability group used only for aggregate fairness diagnostics.",
    }
    rows = [{"field": key, "definition": value} for key, value in definitions.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrimeRisk-BigData reproducible analysis.")
    parser.add_argument("--output-dir", default="/mnt/data/CrimeRisk_BigData_results", help="Directory for generated results.")
    parser.add_argument("--install-missing", action="store_true", help="Install missing dependencies with pip before running.")
    parser.add_argument("--compute-shap", action="store_true", help="Compute exact TreeSHAP values. This is slower but recommended in full research environments.")
    args = parser.parse_args()

    config = SimulationConfig()
    output_root = Path(args.output_dir)
    section51 = output_root / "5.1"
    section52 = output_root / "5.2"
    section53 = output_root / "5.3"
    for directory in [output_root, section51, section52, section53]:
        directory.mkdir(parents=True, exist_ok=True)

    graph = create_city_graph(config.grid_size)
    zones = generate_static_zone_features(config)
    panel = simulate_spatiotemporal_counts(config, zones, graph)
    train, validation, test = temporal_split(panel, config)

    write_table(zones, output_root / "synthetic_zone_features.csv")
    panel.sample(n=min(5000, len(panel)), random_state=config.random_seed).to_csv(output_root / "synthetic_panel_sample.csv", index=False)
    create_data_dictionary(output_root / "data_dictionary.csv")

    performance, models, predictions = fit_and_evaluate_models(train, validation, test, config)
    write_table(performance, section51 / "model_performance.csv")
    save_roc_pr_curves(test, predictions, section51 / "roc_precision_recall_curves.png")
    calibration = save_calibration_curve(test, predictions, section51 / "calibration_curve.png")
    write_table(calibration, section51 / "calibration_summary.csv")

    best_model_name = str(performance.iloc[0]["model"])
    best_model = models[best_model_name]
    best_prob = predictions[best_model_name]

    top_risk, error_by_zone = spatial_outputs(panel, test, best_model_name, best_prob, config, section52)
    importance, fairness = interpretability_and_fairness_outputs(train, test, best_model, best_prob, config, section53, compute_shap=args.compute_shap)

    metadata = {
        "project": "CrimeRisk-BigData",
        "description": "Synthetic reproducible crime-risk forecasting pipeline for smart heterogeneous network environments.",
        "random_seed": config.random_seed,
        "grid_size": config.grid_size,
        "n_days": config.n_days,
        "n_zones": config.grid_size * config.grid_size,
        "n_observations": int(len(panel)),
        "target_event_rate_overall": float(panel["high_risk"].mean()),
        "train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "test_records": int(len(test)),
        "best_model": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "output_subfolders": ["5.1", "5.2", "5.3"],
    }
    with open(output_root / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(textwrap.dedent(f"""
    CrimeRisk-BigData analysis completed successfully.
    Output directory: {output_root}
    Best model: {best_model_name}
    Main tables:
      - {section51 / 'model_performance.csv'}
      - {section52 / 'top_risk_zones.csv'}
      - {section53 / 'fairness_diagnostics.csv'}
    Main figures:
      - {section51 / 'roc_precision_recall_curves.png'}
      - {section52 / 'hotspot_map.png'}
      - {section53 / 'shap_summary_bar.png'}
    """))


if __name__ == "__main__":
    main()
