import contextlib
from copy import deepcopy
import gc
import pathlib
import pickle
import platform
import time
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import wilcoxon

from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Update global parameters for high-quality, readable plots
plt.rcParams.update({
    # 1. Dimensions and Resolution
    "figure.figsize": (6, 4),    # Exactly matches the 6-inch textwidth from automl.sty
    "figure.dpi": 100,           # Standard for screen viewing
    "savefig.dpi": 600,          # High resolution for the final PDF
    "savefig.bbox": "tight",     # Crop whitespace
    "savefig.pad_inches": 0.05,  # Minimal padding

    # 2. Font Consistency (Matching Linux Libertine)
    # The AutoML template uses 'libertine'. We try to find it or a close serif equivalent.
    "font.family": "sans-serif",
    "font.size": 11,             # Match paper's base font size (11pt)
    "axes.titlesize": 14,        # Slightly larger for titles
    "axes.titleweight": "bold",
    "axes.labelsize": 12,        # Match base text
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,       # Slightly smaller for ticks
    "ytick.labelsize": 10,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",

    # 3. Math Rendering (Internal Mathtext)
    # Uses Matplotlib's internal TeX parser without needing a full LaTeX install
    "mathtext.fontset": "stix",  # STIX is a close visual match for Libertine/Times math

    # 4. Lines and Markers (High Visibility)
    "lines.linewidth": 1.5,      # Professional thickness (not too chunky)
    "lines.markersize": 4,
    "axes.linewidth": 1.0,       # Clean bounding box
    "patch.linewidth": 1.5,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.6,           # Subtle grids
    "grid.linestyle": "--",

    # 5. PDF Compatibility
    "pdf.fonttype": 42,          # Use TrueType fonts (required by many conferences)
    "ps.fonttype": 42,           # Avoids Type 3 font errors in PDF checkers

    # 6. Legend Styling
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "legend.framealpha": 0.8,
    "legend.fancybox": False,    # Square corners look more professional in papers
    "legend.shadow": False,
})

class TargetScale(Enum):
    LOG = "log"
    ORIGINAL = "original"
    MAX = "max"

    @classmethod
    def from_str(cls, label: str):
        """Converts a string (from CLI) to an Enum member."""
        mapping = {
            "log": cls.LOG,
            "original": cls.ORIGINAL,
            "max": cls.MAX
        }
        try:
            return mapping[label.lower()]
        except KeyError:
            raise ValueError(f"Invalid target_scale: {label}. Expected one of {list(mapping.keys())}")

# plotting functions for the main experiment
# -----
def plot_ml_heatmap(results_data, visual_config, plot_scenario=None, plot_metric=None, plot_title=None, mark_best=False, layout_grid=None):
    """
    Generates a grid of heatmaps (Rows = Scenarios, Columns = Metrics).
    Tick marks are removed from axes and colorbars for a clean academic look.
    Includes custom grid layout support with auto-sizing (-1) and bottom-row centering.
    """
    # 1. Identify all available entities for filtering
    all_models = list(results_data.keys())
    all_scenarios = []
    all_metrics_found = []
    
    for model_runs in results_data.values():
        if model_runs:
            all_scenarios.extend([run['scenario'] for run in model_runs])
            all_metrics_found.extend(list(model_runs[0]['instance_summary'].keys()))
    
    all_scenarios = sorted(list(set(all_scenarios)))
    all_metrics_found = set(all_metrics_found)

    FIXED_METRIC_ORDER = ["NLLH", "CRPS", "Wasserstein", "KS"]
    
    if plot_metric is not None:
        requested_metrics = [plot_metric] if isinstance(plot_metric, str) else plot_metric
        metrics = sorted(requested_metrics, key=lambda x: FIXED_METRIC_ORDER.index(x) if x in FIXED_METRIC_ORDER else 99)
    else:
        metrics = [m for m in FIXED_METRIC_ORDER if m in all_metrics_found]

    if plot_scenario is not None:
        scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
    else:
        scenarios = all_scenarios

    # --- NEW: Layout Grid Setup (Supporting -1 Auto-Size) ---
    use_custom_grid = layout_grid is not None and (plot_scenario is not None or plot_metric is not None)
    total_plots = len(scenarios) * len(metrics)
    
    if use_custom_grid:
        g_rows, g_cols = layout_grid
        if g_rows == -1: g_rows = int(np.ceil(total_plots / g_cols))
        if g_cols == -1: g_cols = int(np.ceil(total_plots / g_rows))
    else:
        g_rows, g_cols = len(scenarios), len(metrics)

    fig, axes = plt.subplots(g_rows, g_cols, 
                             figsize=(8 * g_cols, 6 * g_rows), 
                             squeeze=False)

    for row_idx, scenario in enumerate(scenarios):
        for col_idx, metric in enumerate(metrics):
            # --- Axis routing ---
            if use_custom_grid:
                plot_idx = row_idx * len(metrics) + col_idx
                ax_row, ax_col = plot_idx // g_cols, plot_idx % g_cols
                ax = axes[ax_row][ax_col]
            else:
                ax_row, ax_col = row_idx, col_idx
                ax = axes[ax_row][ax_col]
            
            # --- AGGREGATION LOGIC ---
            records = []
            for model_name in all_models:
                runs = results_data[model_name]
                for run in runs:
                    if run['scenario'] == scenario:
                        values = run['instance_summary'].get(metric)
                        if values is None: continue
                        run_score = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                        
                        # Check for full-dataset runs and calculate effective context_size
                        ctx_size = run['context_size']
                        is_full_dataset = False
                        
                        if ctx_size is None:
                            ctx_size = int(run['train_size'][0] / run['num_samples_per_instance'])
                            is_full_dataset = True
                        
                        records.append({
                            'context_size': ctx_size,
                            'num_samples_per_instance': run['num_samples_per_instance'],
                            'fold': run['fold'],
                            'seed_ctx': run.get('seed_context_size', -1) if run.get('seed_context_size') is not None else -1,
                            'seed_samp': run['seed_samples_per_instance'],
                            'score': run_score,
                            'is_full': is_full_dataset 
                        })
            
            if not records:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            # Normalize CV Fold Discrepancies
            full_runs = [r for r in records if r['is_full']]
            if full_runs:
                max_ctx_map = {}
                for r in full_runs:
                    ns = r['num_samples_per_instance']
                    if ns not in max_ctx_map or r['context_size'] > max_ctx_map[ns]:
                        max_ctx_map[ns] = r['context_size']
                
                for r in records:
                    if r['is_full']:
                        r['context_size'] = max_ctx_map[r['num_samples_per_instance']]

            df = pd.DataFrame(records)
            df_step2 = df.groupby(['fold', 'seed_ctx', 'context_size', 'num_samples_per_instance'])['score'].mean().reset_index()
            df_step3 = df_step2.groupby(['fold', 'context_size', 'num_samples_per_instance'])['score'].mean().reset_index()
            final_df = df_step3.groupby(['context_size', 'num_samples_per_instance'])['score'].mean().reset_index()
            
            pivot_table = final_df.pivot(index='context_size', columns='num_samples_per_instance', values='score')
            pivot_table = pivot_table.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            cmap = visual_config.get('cmap', 'viridis')
            annot = visual_config.get('annot', True)
            fmt = visual_config.get('fmt', '.4f')
            
            sns.heatmap(
                pivot_table, annot=annot, fmt=fmt, cmap=cmap, 
                cbar_kws={'label': metric}, linewidths=.5, ax=ax,
                mask=pivot_table.isnull()
            )
            
            # --- HIGHLIGHT BEST SCORES ---
            if mark_best and not pivot_table.empty:
                global_min = pivot_table.min().min()
                
                for i, (idx, row) in enumerate(pivot_table.iterrows()):
                    if row.isnull().all():
                        continue
                    
                    local_min = row.min()
                    best_col = row.idxmin()
                    j = pivot_table.columns.get_loc(best_col)
                    
                    x_pos = j + 0.85
                    y_pos = i + 0.85 
                    
                    if local_min == global_min:
                        ax.scatter(x_pos, y_pos, color='gold', marker='*', s=150, 
                                   edgecolor='black', linewidth=0.5, zorder=5)
                    else:
                        ax.scatter(x_pos, y_pos, color='silver', marker='*', s=70, 
                                   edgecolor='black', linewidth=0.5, zorder=5)
            
            # --- REMOVE TICK MARKS ---
            ax.tick_params(axis='both', which='both', length=0)
            
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(axis='both', which='both', length=0)

            ax.invert_yaxis() 
            
            # --- TITLES AND LABELS (Grid Layout) ---
            if len(scenarios) == 1 and len(metrics) == 1:
                # Specific Scenario & Specific Metric -> Title becomes Scenario
                ax.set_title(scenario, pad=20)
                ax.set_ylabel("# Instances")
            elif use_custom_grid:
                if len(metrics) == 1:
                    ax.set_title(scenario, pad=20)
                    ax.set_ylabel("# Instances")
                else:
                    ax.set_title(metric, pad=20)
                    if ax_col == 0: 
                        ax.set_ylabel(f"{scenario}\n# Instances")
                    else:
                        ax.set_ylabel("# Instances")
            else:
                # Standard Layout (Rows = Scenarios, Cols = Metrics)
                if row_idx == 0:
                    ax.set_title(metric, pad=20)
                    
                if col_idx == 0:
                    ax.set_ylabel(f"{scenario}\n# Instances")
                else:
                    ax.set_ylabel("# Instances")
                
            ax.set_xlabel("# Samples per Instance")

    # --- NEW: Hide (don't delete) unused subplots to preserve GridSpec structure ---
    if use_custom_grid:
        for i in range(total_plots, g_rows * g_cols):
            axes[i // g_cols][i % g_cols].set_visible(False)

    if plot_title:
        fig.suptitle(plot_title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()
        
    # --- NEW: Center the orphaned heatmaps in the last row ---
    if use_custom_grid:
        remainder = total_plots % g_cols
        if remainder != 0 and g_cols > 1:
            # 1. Calculate physical screen distance of a single column jump
            pos0 = axes[0][0].get_position()
            pos1 = axes[0][1].get_position()
            col_stride = pos1.x0 - pos0.x0
            
            # 2. Calculate shift required to center
            shift = (g_cols - remainder) * col_stride / 2.0
            last_row_idx = g_rows - 1
            
            # 3. Apply manual positional shift to BOTH heatmap and colorbar
            for j in range(remainder):
                ax_to_shift = axes[last_row_idx][j]
                
                # Shift the main heatmap
                pos = ax_to_shift.get_position()
                pos.x0 += shift
                pos.x1 += shift
                ax_to_shift.set_position(pos)
                
                # Shift the associated colorbar (if it exists)
                if len(ax_to_shift.collections) > 0:
                    cbar = ax_to_shift.collections[0].colorbar
                    if cbar is not None:
                        cbar_pos = cbar.ax.get_position()
                        cbar_pos.x0 += shift
                        cbar_pos.x1 += shift
                        cbar.ax.set_position(cbar_pos)
        
    return fig

def plot_heatmap_scatter(results_data, visual_config, n_instances=None, n_samples_per_instance=None, 
                         overlay=False, log_x=False, log_y=False, zoom_on_model=None, plot_title=None, 
                         plot_scenario=None, plot_metric=None,
                         show_legend=True, global_legend=False, use_hatching=False, layout_grid=None):
    """
    Generates styled line/scatter plots with variance shading derived from heatmap data.
    Aligned with the styling and layout engine of plot_full_instances_results.
    """
    # 1. Validation
    if (n_instances is None) == (n_samples_per_instance is None):
        raise ValueError("Exactly one of 'n_instances' or 'n_samples_per_instance' must be provided.")
        
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []
    all_scenarios = []
    all_metrics_found = []
    
    for model_runs in results_data.values():
        if model_runs:
            all_scenarios.extend([run['scenario'] for run in model_runs])
            all_metrics_found.extend(list(model_runs[0]['instance_summary'].keys()))
    
    all_scenarios = sorted(list(set(all_scenarios)))
    all_metrics_found = set(all_metrics_found)

    FIXED_METRIC_ORDER = ["NLLH", "CRPS", "Wasserstein", "KS"]
    if plot_metric is not None:
        requested_metrics = [plot_metric] if isinstance(plot_metric, str) else plot_metric
        metrics = sorted(requested_metrics, key=lambda x: FIXED_METRIC_ORDER.index(x) if x in FIXED_METRIC_ORDER else 99)
    else:
        metrics = [m for m in FIXED_METRIC_ORDER if m in all_metrics_found]

    scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
    scenarios = scenarios if scenarios is not None else all_scenarios

    for model_name, runs in results_data.items():
        for scenario in scenarios:
            scen_runs = [r for r in runs if r['scenario'] == scenario]
            if not scen_runs: continue
            
            scen_records = []
            for run in scen_runs:
                ctx_size = run['context_size']
                is_full_dataset = False
                if ctx_size is None:
                    ctx_size = int(run['train_size'][0] / run['num_samples_per_instance'])
                    is_full_dataset = True
                    
                for metric in metrics:
                    values = run['instance_summary'].get(metric)
                    if values is None: continue
                    run_score = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                    
                    scen_records.append({
                        "model": model_name,
                        "scenario": scenario,
                        "metric": metric,
                        "context_size": ctx_size,
                        "num_samples_per_instance": run['num_samples_per_instance'],
                        "fold": run['fold'],
                        "seed_ctx": run.get('seed_context_size', -1) if run.get('seed_context_size') is not None else -1,
                        "score": run_score,
                        "is_full": is_full_dataset
                    })
            
            # Normalize CV Fold Discrepancies
            full_runs = [r for r in scen_records if r['is_full']]
            if full_runs:
                max_ctx_map = {}
                for r in full_runs:
                    ns = r['num_samples_per_instance']
                    if ns not in max_ctx_map or r['context_size'] > max_ctx_map[ns]:
                        max_ctx_map[ns] = r['context_size']
                for r in scen_records:
                    if r['is_full']:
                        r['context_size'] = max_ctx_map[r['num_samples_per_instance']]
                        
            all_records.extend(scen_records)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No data found to plot.")

    # Apply axis filtering based on user input
    if n_instances is not None:
        x_col = 'num_samples_per_instance'
        x_label = "# Samples per Instance"
        overlay_col = 'context_size'
        overlay_suffix = "Instances"
        
        if not overlay:
            filtered_dfs = []
            for scen in df['scenario'].unique():
                scen_df = df[df['scenario'] == scen]
                target_instances = min(n_instances, scen_df['context_size'].max())
                filtered_dfs.append(scen_df[scen_df['context_size'] == target_instances])
            df = pd.concat(filtered_dfs) if filtered_dfs else pd.DataFrame()
    else:
        x_col = 'context_size'
        x_label = "Context Size"
        overlay_col = 'num_samples_per_instance'
        overlay_suffix = "Samples/Inst"
        
        if not overlay:
            df = df[df['num_samples_per_instance'] == n_samples_per_instance]

    if df.empty:
        raise ValueError("No data found after filtering for the specified instances/samples.")

    # --- Build dynamic loop configuration & overlay gradients ---
    if overlay:
        model_to_use = list(results_data.keys())[0] # Assumes single model
        overlay_vals = sorted(df[overlay_col].unique())
        
        base_color = visual_config[model_to_use][0]
        # Generate gradient, dropping the first white color
        colors = sns.light_palette(base_color, n_colors=len(overlay_vals) + 1)[1:]
            
        plot_items = []
        for i, val in enumerate(overlay_vals):
            plot_items.append({
                'label': f"{val} {overlay_suffix}",
                'color': colors[i],
                'hatch': '',
                'filter_model': model_to_use,
                'filter_val': val
            })
    else:
        plot_items = []
        for m_name, (color, hatch) in visual_config.items():
            plot_items.append({
                'label': m_name,
                'color': color,
                'hatch': hatch,
                'filter_model': m_name,
                'filter_val': None
            })

    # --- Grid Layout Setup ---
    use_custom_grid = layout_grid is not None and (plot_scenario is not None or plot_metric is not None)
    if use_custom_grid:
        g_rows, g_cols = layout_grid
    else:
        g_rows, g_cols = len(scenarios), len(metrics)

    fig, axes = plt.subplots(g_rows, g_cols, 
                             figsize=(6 * g_cols, 5 * g_rows), 
                             squeeze=False)

    model_handles = {}
    special_handles = {}
    has_out_of_bounds = False

    for row_idx, scenario in enumerate(scenarios):
        scenario_df = df[df['scenario'] == scenario]
        if scenario_df.empty:
            continue
            
        scenario_x_sizes = sorted(scenario_df[x_col].unique())
        
        for col_idx, metric in enumerate(metrics):
            # --- Axis routing ---
            if use_custom_grid:
                plot_idx = row_idx * len(metrics) + col_idx
                ax_row, ax_col = plot_idx // g_cols, plot_idx % g_cols
                ax = axes[ax_row][ax_col]
            else:
                ax_row, ax_col = row_idx, col_idx
                ax = axes[ax_row][ax_col]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            line_objs = {}
            mini_leg = None
            
            for p_item in plot_items:
                lbl = p_item['label']
                color = p_item['color']
                hatch = p_item['hatch']
                model_name = p_item['filter_model']
                
                subset = scenario_df[(scenario_df['model'] == model_name) & 
                                     (scenario_df['metric'] == metric)]
                
                if overlay:
                    subset = subset[subset[overlay_col] == p_item['filter_val']]
                    
                if subset.empty:
                    continue

                df_step2 = subset.groupby(['fold', 'seed_ctx', x_col])['score'].mean().reset_index()
                df_step3 = df_step2.groupby(['fold', x_col])['score'].mean().reset_index()
                final_stats = df_step3.groupby(x_col)['score'].agg(['mean', 'std']).reset_index()
                final_stats = final_stats.sort_values(x_col)
                
                x = final_stats[x_col]
                y_mean = final_stats['mean']
                y_std = final_stats['std'].fillna(0)
                
                line, = ax.plot(x, y_mean, label=lbl, color=color, marker='o')
                line_objs[lbl] = line
                if lbl not in model_handles:
                    model_handles[lbl] = line
                
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                if log_y and metric != 'NLLH':
                    lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                
                hatch_val = hatch if use_hatching else None
                ax.fill_between(x, lower_bound, upper_bound, 
                                color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')

                if zoom_on_model == model_name:
                    lower_bound_zoom = (y_mean - y_std).min()
                    upper_bound_zoom = (y_mean + y_std).max()
                    zoom_bounds["min"] = min(zoom_bounds["min"], lower_bound_zoom)
                    zoom_bounds["max"] = max(zoom_bounds["max"], upper_bound_zoom)

            if log_x:
                ax.set_xscale('log')
                ax.set_xticks(scenario_x_sizes) 
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            if log_y and metric != 'NLLH':
                ax.set_yscale('log')

            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and metric != 'NLLH' and low <= 0:
                    low = zoom_bounds["min"] * 0.9 if zoom_bounds["min"] > 0 else 1e-9
                ax.set_ylim(low, high)

            # Handle Invisible Models (Mini-Legend Style)
            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                out_of_bounds_models = []
                for lbl, line in line_objs.items():
                    p_cfg = next(item for item in plot_items if item['label'] == lbl)
                    subset = scenario_df[(scenario_df['model'] == p_cfg['filter_model']) & 
                                         (scenario_df['metric'] == metric)]
                    if overlay:
                        subset = subset[subset[overlay_col] == p_cfg['filter_val']]
                        
                    if subset.empty: continue
                    
                    df_step2 = subset.groupby(['fold', 'seed_ctx', x_col])['score'].mean().reset_index()
                    df_step3 = df_step2.groupby(['fold', x_col])['score'].mean().reset_index()
                    stats = df_step3.groupby(x_col)['score'].agg(['mean', 'std'])
                    
                    if (stats['mean'] - stats['std'].fillna(0)).min() > current_ylim_max:
                        out_of_bounds_models.append(lbl)
                
                if out_of_bounds_models:
                    has_out_of_bounds = True
                    if show_legend:
                        oob_handles = [plt.Line2D([], [], color='dimgray', linestyle='none', 
                                                  marker='$\\gg$', markeredgecolor='dimgray', 
                                                  markeredgewidth=0.5, markersize=10) for m in out_of_bounds_models]
                        mini_leg = ax.legend(oob_handles, out_of_bounds_models, loc='lower right')
                        mini_leg.set_zorder(10)

            # --- Layout / Titling ---
            if use_custom_grid:
                if len(metrics) == 1:
                    ax.set_title(scenario)
                    if ax_col == 0: ax.set_ylabel(metric)
                else:
                    ax.set_title(metric)
                    if ax_col == 0: ax.set_ylabel(scenario)
            else:
                if row_idx == 0: ax.set_title(f"Metric: {metric}")
                ax.set_ylabel(f"{scenario}\n\n{metric}" if col_idx == 0 else metric)
            
            ax.set_xlabel(x_label)
            ax.grid(True)
            
            # --- Local Legend Handling ---
            if show_legend and not global_legend:
                main_leg = ax.legend(loc='upper right')
                if mini_leg:
                    # Matplotlib drops the first legend if you call ax.legend() twice. We safely add it back.
                    ax.add_artist(mini_leg)

    # --- Clean up unused subplots ---
    if use_custom_grid:
        total_plots = len(scenarios) * len(metrics)
        for i in range(total_plots, g_rows * g_cols):
            fig.delaxes(axes[i // g_cols][i % g_cols])

    # --- Global Legend & Spacing Setup ---
    if has_out_of_bounds and global_legend:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='dimgray', linewidths=0.5, s=100)

    final_handles = list(model_handles.values()) + list(special_handles.values())
    final_labels = list(model_handles.keys()) + list(special_handles.keys())
    ncols = len(final_handles)
    
    fig_height = 5 * g_rows

    # Setup the spacing and titles depending on the global_legend boolean
    if show_legend and global_legend:
        if plot_title:
            top_space = 1.0
            legend_y = 1.0 - (0.5 / fig_height)
            title_y = 1.0 - (0.15 / fig_height)
            fig.suptitle(plot_title, y=title_y)
        else:
            top_space = 0.7
            legend_y = 1.0 - (0.15 / fig_height)
            
        fig.legend(final_handles, final_labels, loc='upper center', 
               bbox_to_anchor=(0.5, legend_y), ncol=ncols)
    else:
        if plot_title:
            top_space = 0.5
            title_y = 1.0 - (0.15 / fig_height)
            fig.suptitle(plot_title, y=title_y)
        else:
            top_space = 0.1
            
    rect_top = 1.0 - (top_space / fig_height)
    plt.tight_layout(rect=[0, 0, 1, rect_top])
    
    plt.close(fig) # Prevent double-print in Jupyter
    return fig

def plot_feat_dropping_results(results_data, visual_config, log_x=False, log_y=False, 
                    zoom_on_model=None, mark_best=False, 
                    plot_scenario=None, plot_metric=None, plot_title=None, grid_layout=None, use_hatching=False):
    """
    Plots ML experiment results for the 'feature dropping' experiment.
    
    Aggregation Logic:
    1. Run-level: Mean of internal instance scores.
    2. Context-Seed-level: Mean across seed_context_size.
    3. Drop-Rate-Seed-level: Mean across seed_feature_drop_rate.
    4. Fold-level: Mean and Std across folds.
    """
    # 1. Validation
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []
    metrics_list = []

    for model_name, runs in results_data.items():
        for run in runs:
            scenario = run['scenario']
            # The x-axis for this experiment
            n_features_keep = run['n_features_keep'] 
            fold = run['fold']
            seed_ctx = run['seed_context_size']
            seed_drop = run['seed_feature_drop_rate']
            summary = run['instance_summary']
            
            for metric_name, values in summary.items():
                if metric_name not in metrics_list:
                    metrics_list.append(metric_name)
                
                # Step 1: Compute the average performance per run (Instance Mean)
                val = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "n_features_keep": n_features_keep, "fold": fold, 
                    "seed_ctx": seed_ctx, "seed_drop": seed_drop, 
                    "metric": metric_name, "value": val
                })

    df = pd.DataFrame(all_records)
    
    # Filtering
    all_available_scenarios = sorted(df['scenario'].unique())
    all_available_metrics = metrics_list

    # Handle Scenario filtering (supports single string or list of strings)
    if plot_scenario is not None:
        # Normalize input to a list
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        
        # Validate that all requested scenarios exist in the available data
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    if plot_metric is not None:
        if plot_metric not in all_available_metrics:
            raise ValueError(f"Metric '{plot_metric}' not found in data.")
        metrics = [plot_metric]
    else:
        metrics = all_available_metrics

    x_ticks = sorted(df['n_features_keep'].unique())

    # 3. Plotting Setup
    n_scenarios, n_metrics = len(scenarios), len(metrics)
    total_plots = n_scenarios * n_metrics
    use_custom_grid = grid_layout is not None and (n_scenarios == 1 or n_metrics == 1)

    if use_custom_grid:
        g_rows, g_cols = grid_layout
        if g_rows == -1: g_rows = int(np.ceil(total_plots / g_cols))
        if g_cols == -1: g_cols = int(np.ceil(total_plots / g_rows))
    else:
        g_rows, g_cols = n_scenarios, n_metrics

    fig, axes = plt.subplots(g_rows, g_cols, 
                             figsize=(6 * g_cols, 5 * g_rows), 
                             squeeze=False)

    model_handles = {}
    special_handles = {}
    has_out_of_bounds = False

    for row_idx, scenario in enumerate(scenarios):
        for col_idx, metric in enumerate(metrics):
            if use_custom_grid:
                plot_idx = row_idx * n_metrics + col_idx
                ax_row, ax_col = plot_idx // g_cols, plot_idx % g_cols
                ax = axes[ax_row][ax_col]
            else:
                ax_row, ax_col = row_idx, col_idx
                ax = axes[ax_row][ax_col]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            
            for model_name, (color, hatch) in visual_config.items():
                subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                if subset.empty: continue

                # --- NESTED AGGREGATION LOGIC ---
                # Step 2: Average across all seed_context_size seeds for each (fold, seed_feature_drop_rate)
                ctx_agg = subset.groupby(['n_features_keep', 'fold', 'seed_drop'])['value'].mean().reset_index()
                
                # Step 3: Average across seed_feature_drop_rate values for each fold
                fold_agg = ctx_agg.groupby(['n_features_keep', 'fold'])['value'].mean().reset_index()
                
                # Step 4: Aggregate across folds (Mean and Std)
                final_stats = fold_agg.groupby('n_features_keep')['value'].agg(['mean', 'std']).reset_index().sort_values('n_features_keep')
                
                x = final_stats['n_features_keep']
                y_mean = final_stats['mean']
                y_std = final_stats['std']
                
                line, = ax.plot(x, y_mean, label=model_name, color=color, marker='o')
                line_objs[model_name] = line
                if model_name not in model_handles:
                    model_handles[model_name] = line
                
                # Conditional Shade Clipping (Prevents Log-Pillars)
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                if log_y and metric != 'NLLH':
                    lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                
                hatch_val = hatch if use_hatching else None
                ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                
                if y_mean.min() < abs_best_val:
                    abs_best_val = y_mean.min()
                    abs_best_model = model_name
                    abs_best_coords = (x[y_mean.idxmin()], abs_best_val)

                if zoom_on_model == model_name:
                    zoom_bounds["min"] = min(zoom_bounds["min"], (y_mean - y_std).min())
                    zoom_bounds["max"] = max(zoom_bounds["max"], (y_mean + y_std).max())

            # Mark Best
            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                sc_best = ax.scatter(bx, by, color='gold', marker='*', s=150, edgecolor='black', linewidth=0.5, zorder=5, label='Best Model')
                line_objs[abs_best_model].set_label(f"{line_objs[abs_best_model].get_label()} <-- Best")
                if 'Best Model' not in special_handles:
                    special_handles['Best Model'] = sc_best

            # Robust Zoom Logic
            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and metric != 'NLLH' and low <= 0:
                    low = zoom_bounds["min"] * 0.9 if zoom_bounds["min"] > 0 else 1e-9
                ax.set_ylim(low, high)

            # Axis Formatting
            current_x_ticks = sorted(subset['n_features_keep'].unique())

            if log_x:
                ax.set_xscale('symlog', linthresh=1)
                ax.xaxis.set_major_formatter(ScalarFormatter())
            
            # 1. Keep ALL ticks visible
            ax.set_xticks(current_x_ticks) 
            
            # 2. Generate labels: Every second value, starting from the LEFT (the max value)
            # Since the axis is inverted, the leftmost value is the last index of current_x_ticks
            all_labels = [str(val) for val in current_x_ticks]
            # We iterate backwards from the end to ensure the leftmost label is always shown
            filtered_labels = [
                all_labels[i] if (len(all_labels) - 1 - i) % 2 == 0 else "" 
                for i in range(len(all_labels))
            ]
            ax.set_xticklabels(filtered_labels)

            # 3. Formatting for readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            
            # 4. Tighten limits and Invert
            ax.set_xlim(min(current_x_ticks), max(current_x_ticks))
            ax.invert_xaxis() 
            if log_y and metric != 'NLLH': ax.set_yscale('log')

            # Handle Invisible Models (Legend Annotation)
            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                for model_name, line in line_objs.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                    if subset.empty: continue
                    # Use the same aggregation sequence to find if the model is off-chart
                    ctx_agg = subset.groupby(['n_features_keep', 'fold', 'seed_drop'])['value'].mean().reset_index()
                    fold_agg = ctx_agg.groupby(['n_features_keep', 'fold'])['value'].mean().reset_index()
                    stats = fold_agg.groupby('n_features_keep')['value'].agg(['mean', 'std'])
                    if (stats['mean'] - stats['std']).min() > current_ylim_max:
                        lbl = line.get_label()
                        if " <-- Best" not in lbl: line.set_label(f"{lbl} (≫)")
                        has_out_of_bounds = True

            # Layout
            if use_custom_grid:
                if n_metrics == 1:
                    ax.set_title(scenario)
                    if ax_col == 0: ax.set_ylabel(metric)
                else:
                    ax.set_title(metric)
                    if ax_col == 0: ax.set_ylabel(scenario)
            else:
                if row_idx == 0: ax.set_title(f"Metric: {metric}")
                ax.set_ylabel(f"{scenario}\n{metric}" if col_idx == 0 else metric)
            
            ax.set_xlabel("# Features")
            ax.grid(True)

    # Delete any unused subplots in a custom grid completely
    if use_custom_grid:
        for i in range(total_plots, g_rows * g_cols):
            fig.delaxes(axes[i // g_cols][i % g_cols])

    # Handle the Out-of-Bounds special legend entry
    if has_out_of_bounds:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='black', linewidths=0.5, s=100)

    # Merge handles, ensuring models are listed first
    final_handles = list(model_handles.values()) + list(special_handles.values())
    final_labels = list(model_handles.keys()) + list(special_handles.keys())

    ncols = len(final_handles)
    fig_height = 5 * g_rows

    if plot_title:
        top_space = 1.6
        legend_y = 1.0 - (0.5 / fig_height)
        title_y = 1.0 - (0.15 / fig_height)
        fig.suptitle(plot_title, y=title_y)
    else:
        top_space = 1.3
        legend_y = 1.0 - (0.15 / fig_height)
        
    rect_top = 1.0 - (top_space / fig_height)

    fig.legend(final_handles, final_labels, loc='upper center', 
               bbox_to_anchor=(0.5, legend_y), ncol=ncols)

    plt.tight_layout(rect=[0, 0, 1, rect_top])
    
    return fig

def plot_full_instances_results(results_data, visual_config, log_x=False, log_y=False, 
                    zoom_on_model=None, mark_best=False, 
                    plot_scenario=None, plot_metric=None, plot_title=None, 
                    show_legend=True, use_hatching=False, layout_grid=None):
    """
    Plots ML experiment results for the 'num_samples_per_instance' experiment.
    Includes all previous features: log-scaling, robust zooming, best-point 
    highlighting, visibility annotations, conditional shade clipping, and custom grid layouts.
    """
    # 1. Validation
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []
    metrics_list = []

    for model_name, runs in results_data.items():
        for run in runs:
            scenario = run['scenario']
            x_val = run['num_samples_per_instance']
            fold = run['fold']
            seed = run['seed_samples_per_instance']
            summary = run['instance_summary']
            
            for metric_name, values in summary.items():
                if metric_name != "RMSE" and metric_name not in metrics_list:
                    metrics_list.append(metric_name)
                
                # Step 1: Mean of internal score array
                val = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "x_val": x_val, "fold": fold, "seed": seed, 
                    "metric": metric_name, "value": val
                })

    df = pd.DataFrame(all_records)
    
    # Filtering for scenarios and metrics
    all_available_scenarios = sorted(df['scenario'].unique())
    all_available_metrics = metrics_list

    # Handle Scenario filtering
    if plot_scenario is not None:
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    if plot_metric is not None:
        if plot_metric not in all_available_metrics:
            raise ValueError(f"Metric '{plot_metric}' not found in data.")
        metrics = [plot_metric]
    else:
        metrics = all_available_metrics

    x_ticks = sorted(df['x_val'].unique())

    # --- NEW: Layout Grid Setup ---
    use_custom_grid = layout_grid is not None and (plot_scenario is not None or plot_metric is not None)
    if use_custom_grid:
        g_rows, g_cols = layout_grid
    else:
        g_rows, g_cols = len(scenarios), len(metrics)

    # 3. Plotting Setup
    fig, axes = plt.subplots(g_rows, g_cols, 
                             figsize=(6 * g_cols, 5 * g_rows), 
                             squeeze=False)

    model_handles = {}
    special_handles = {}
    has_out_of_bounds = False

    for row_idx, scenario in enumerate(scenarios):
        scenario_max_x = df[df['scenario'] == scenario]['x_val'].max()
        
        for col_idx, metric in enumerate(metrics):
            # --- NEW: Axis routing ---
            if use_custom_grid:
                plot_idx = row_idx * len(metrics) + col_idx
                ax_row, ax_col = plot_idx // g_cols, plot_idx % g_cols
                ax = axes[ax_row][ax_col]
            else:
                ax_row, ax_col = row_idx, col_idx
                ax = axes[ax_row][ax_col]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            added_early_term_legend = False
            
            for model_name, (color, hatch) in visual_config.items():
                subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                if subset.empty: continue

                # --- AGGREGATION LOGIC ---
                seed_counts = subset.groupby(['x_val', 'fold']).size()
                if not seed_counts.eq(5).all():
                    raise AssertionError(f"Model {model_name} in {scenario} does not have exactly 5 seeds for all x_val/folds.")
                
                fold_means = subset.groupby(['x_val', 'fold'])['value'].mean().reset_index()
                final_stats = fold_means.groupby('x_val')['value'].agg(['mean', 'std']).reset_index().sort_values('x_val')
                
                x = final_stats['x_val']
                y_mean = final_stats['mean']
                y_std = final_stats['std']
                
                # Plot line
                line, = ax.plot(x, y_mean, label=model_name, color=color, marker='o')
                line_objs[model_name] = line
                if model_name not in model_handles:
                    model_handles[model_name] = line
                
                # Conditional Shade Clipping (Prevents Log-Pillars)
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                if log_y and metric != 'NLLH':
                    lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                
                hatch_val = hatch if use_hatching else None
                ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                
                # --- EARLY TERMINATION MARKER (Restored) ---
                current_max_x = x.max()
                if current_max_x < scenario_max_x:
                    last_x = final_stats['x_val'].iloc[-1]
                    last_y = final_stats['mean'].iloc[-1]
                    
                    # Toned down the offsets so they don't jump too far right on large data ranges
                    if log_x:
                        x_offset = last_x * 1.06  
                    else:
                        x_range = scenario_max_x - x_ticks[0]
                        x_offset = last_x + (x_range * 0.01) 
                        
                    label = 'Timeout' if not added_early_term_legend else '_nolegend_'
                    sc = ax.scatter(x_offset, last_y, color=color, marker='x', s=60, 
                               linewidth=1.5, zorder=6, label=label)
                    added_early_term_legend = True
                    if 'Timeout' not in special_handles: 
                        special_handles['Timeout'] = sc
                # ---------------------------------------------------
                
                # Track Best
                if y_mean.min() < abs_best_val:
                    abs_best_val = y_mean.min()
                    abs_best_model = model_name
                    abs_best_coords = (x[y_mean.idxmin()], abs_best_val)

                # Update Zoom Bounds
                if zoom_on_model == model_name:
                    zoom_bounds["min"] = min(zoom_bounds["min"], (y_mean - y_std).min())
                    zoom_bounds["max"] = max(zoom_bounds["max"], (y_mean + y_std).max())

            # Mark Best
            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                sc_best = ax.scatter(bx, by, color='gold', marker='*', s=150, edgecolor='black', linewidth=0.5, zorder=5, label='Best Model')
                if 'Best Model' not in special_handles:
                    special_handles['Best Model'] = sc_best

            # Robust Zoom Logic
            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and metric != 'NLLH' and low <= 0:
                    low = zoom_bounds["min"] * 0.9 if zoom_bounds["min"] > 0 else 1e-9
                ax.set_ylim(low, high)

            # Axis Formatting
            if log_x:
                ax.set_xscale('log')
                ax.set_xticks(x_ticks) 
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if log_y and metric != 'NLLH': ax.set_yscale('log')

            # Handle Invisible Models
            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                out_of_bounds_models = []
                for model_name, line in line_objs.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                    if subset.empty: continue
                    fold_means = subset.groupby(['x_val', 'fold'])['value'].mean().reset_index()
                    stats = fold_means.groupby('x_val')['value'].agg(['mean', 'std'])
                    if (stats['mean'] - stats['std']).min() > current_ylim_max:
                        out_of_bounds_models.append(model_name)
                        
                if out_of_bounds_models:
                    has_out_of_bounds = True
                    if show_legend:
                        oob_handles = [plt.Line2D([], [], color='dimgray', linestyle='none', 
                                                  marker='$\\gg$', markeredgecolor='dimgray', 
                                                  markeredgewidth=0.5, markersize=10) for m in out_of_bounds_models]
                        mini_leg = ax.legend(oob_handles, out_of_bounds_models, loc='upper right')
                        mini_leg.set_zorder(10)

            # --- NEW: Layout ---
            if len(scenarios) == 1 and len(metrics) == 1:
                # Case 1: Specific Scenario & Specific Metric -> Title becomes Scenario
                ax.set_title(scenario)
                ax.set_ylabel(metric)
            elif use_custom_grid:
                if n_metrics == 1:
                    ax.set_title(scenario)
                    if ax_col == 0: ax.set_ylabel(metric)
                else:
                    ax.set_title(metric)
                    if ax_col == 0: ax.set_ylabel(scenario)
            else:
                if row_idx == 0:
                    ax.set_title(f"Metric: {metric}")
                
                if col_idx == 0:
                    ax.set_ylabel(f"{scenario}\n\n{metric}")
                else:
                    ax.set_ylabel(metric)
                
            ax.set_xlabel("# Samples per Instance")
            ax.grid(True)

    # --- NEW: Clean up unused subplots ---
    if use_custom_grid:
        total_plots = len(scenarios) * len(metrics)
        for i in range(total_plots, g_rows * g_cols):
            fig.delaxes(axes[i // g_cols][i % g_cols])

    # 4. Global Legend & Spacing Setup
    if has_out_of_bounds:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='dimgray', linewidths=0.5, s=100)

    final_handles = list(model_handles.values()) + list(special_handles.values())
    final_labels = list(model_handles.keys()) + list(special_handles.keys())
    ncols = len(final_handles)
    
    fig_height = 5 * g_rows # <-- NEW: Fixed spacing calculation

    # Reclaim whitespace if legends are toggled off
    if show_legend:
        if plot_title:
            top_space = 1.0
            legend_y = 1.0 - (0.5 / fig_height)
            title_y = 1.0 - (0.15 / fig_height)
            fig.suptitle(plot_title, y=title_y)
        else:
            top_space = 0.7
            legend_y = 1.0 - (0.15 / fig_height)
            
        fig.legend(final_handles, final_labels, loc='upper center', 
               bbox_to_anchor=(0.5, legend_y), ncol=ncols)
    else:
        if plot_title:
            top_space = 0.5
            title_y = 1.0 - (0.15 / fig_height)
            fig.suptitle(plot_title, y=title_y)
        else:
            top_space = 0.1
            
    rect_top = 1.0 - (top_space / fig_height)
    plt.tight_layout(rect=[0, 0, 1, rect_top])
    
    return fig

def plot_instance_level_bar_chart(results_data, context_size, metric_name, plot_title=None, lower_is_better=True):
    """
    Plots an Instance-Level Stacked Bar Chart of Rank Proportions.
    Calculates consistency by ranking models on every single test instance independently.
    
    Args:
        results_data: Dict of {model_name: list_of_runs}
        context_size: Int, the specific context size to filter the data by
        metric_name: String, the specific metric to evaluate
        plot_title: String, optional custom title
        lower_is_better: Boolean, True for metrics like Loss, False for Accuracy
        
    Returns:
        matplotlib Figure object
    """
    if not results_data:
        raise ValueError("results_data is empty.")

    # 1. Group data by (scenario, fold) and compute seed-averaged instance arrays
    extracted_data = {}
    
    for model_name, runs in results_data.items():
        seeds_dict = {} 
        for run in runs:
            if run['context_size'] != context_size:
                continue
            if metric_name not in run['instance_summary']:
                continue
            
            scenario = run['scenario']
            fold = run['fold']
            values = run['instance_summary'][metric_name]
            
            arr = values.numpy() if hasattr(values, 'numpy') else np.array(values)
            
            key = (scenario, fold)
            if key not in seeds_dict:
                seeds_dict[key] = []
            seeds_dict[key].append(arr)
        
        for key, arrays in seeds_dict.items():
            seed_avg_arr = np.mean(arrays, axis=0) 
            
            if key not in extracted_data:
                extracted_data[key] = {}
            extracted_data[key][model_name] = seed_avg_arr

    if not extracted_data:
        raise ValueError(f"No data found for context_size={context_size} and metric={metric_name}")

    # 2. Instance-Level Ranking
    all_ranks = []
    
    for key, model_arrays in extracted_data.items():
        available_models = list(model_arrays.keys())
        
        matrix = np.stack([model_arrays[m] for m in available_models]).T
        df_instances = pd.DataFrame(matrix, columns=available_models)
        
        ranks = df_instances.rank(axis=1, method='min', ascending=lower_is_better).astype(int)
        
        for m in available_models:
            for r in ranks[m]:
                all_ranks.append({'model': m, 'rank': r})

    # 3. Calculate Proportions & Stats
    df_ranks = pd.DataFrame(all_ranks)
    unique_models_in_data = df_ranks['model'].unique()
    
    # ADJUSTMENT: Calculate the precise average rank per model
    avg_ranks = df_ranks.groupby('model')['rank'].mean()
    
    rank_counts = df_ranks.groupby(['model', 'rank']).size().unstack(fill_value=0)
    rank_counts = rank_counts.reindex(index=unique_models_in_data, fill_value=0)
    
    # Calculate total instances (N) for the title
    total_instances = int(rank_counts.sum(axis=1).iloc[0])
    
    max_possible_rank = len(unique_models_in_data)
    for r in range(1, max_possible_rank + 1):
        if r not in rank_counts.columns:
            rank_counts[r] = 0
    rank_counts = rank_counts.reindex(columns=sorted(rank_counts.columns), fill_value=0)

    rank_props = rank_counts.div(rank_counts.sum(axis=1), axis=0) * 100

    sort_cols = sorted(rank_props.columns)
    rank_props = rank_props.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    models = list(rank_props.index)

    # 4. Plotting Setup
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    
    num_ranks = len(rank_counts.columns)
    fallback_cmap = plt.get_cmap("Reds")
    rank_colors = {}
    
    for rank in sorted(rank_counts.columns):
        if rank == 1:
            rank_colors[rank] = '#2CA02C' 
        elif rank == 2:
            rank_colors[rank] = '#FF7F0E' 
        else:
            if num_ranks > 2:
                ratio = (rank - 3) / max(1, num_ranks - 3)
                rank_colors[rank] = fallback_cmap(0.3 + 0.5 * ratio) 
            else:
                rank_colors[rank] = '#d62728'

    # 5. Draw Stacked Bars
    for x_idx, model in enumerate(models):
        bottom = 0
        model_props = rank_props.loc[model]
        
        for rank in sorted(rank_counts.columns):
            prop = model_props[rank]
            if prop > 0:
                ax.bar(x_idx, prop, bottom=bottom, color=rank_colors[rank], 
                       edgecolor='white', width=0.3)
                bottom += prop

    # 6. Formatting
    ax.set_xticks(range(len(models)))
    
    # Set the model names and append a newline to reserve vertical space
    ax.set_xticklabels([f"{m}\n" for m in models], rotation=0)
    
    # Draw the smaller Mean Rank text directly into that reserved space
    for x_idx, m in enumerate(models):
        ax.text(x_idx, -0.06, f"Mean Rank: {avg_ranks[m]:.2f}", 
            transform=ax.get_xaxis_transform(), 
            ha='center', va='top')
    
    ax.set_ylabel("Percentage of Total Instances (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y')
    
    # ADJUSTMENT: Inject total_instances (N) into the title
    if plot_title:
        ax.set_title(plot_title, pad=40)
    # else:
    #     ax.set_title(
    #         f"Instance-Level Rank Proportions\n"
    #         f"(Metric: {metric_name} | Context Size: {context_size} | N = {total_instances:,})", 
    #         fontsize=14, fontweight='bold', pad=40
    #     )

    rank_patches = [mpatches.Patch(facecolor=rank_colors[r], edgecolor='white', label=f'Rank {r}') 
                    for r in sorted(rank_counts.columns)]
    
    ax.legend(handles=rank_patches, loc='lower center', bbox_to_anchor=(0.5, 1.02), fontsize=10,
              ncol=len(rank_counts.columns))

    plt.tight_layout()
    return fig

def plot_time_results(results_data, visual_config, time_of=None, log_x=False, log_y=False, 
                      zoom_on_model=None, plot_title=None, mark_best=False, 
                      plot_scenario=None, use_hatching=False):
    """
    Plots ML experiment wall-clock time results (fit and/or predict) and tradeoff curves.
    
    Args:
        results_data: Dict of {model_name: list_of_runs}
        visual_config: Dict of {model_name: (color, hatch)}
        time_of: String ('fit', 'predict') or None (plots both side-by-side)
        log_x, log_y: Booleans for axis scaling
        zoom_on_model: Model name to base y-axis limits on
        mark_best: Boolean to highlight the best (lowest) point with a golden star
        plot_scenario: String or list, if provided, only plot specific scenarios
    """
 
    # 1. Validation
    if time_of not in [None, 'fit', 'predict']:
        raise ValueError("time_of must be None, 'fit', or 'predict'")
        
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []

    for model_name, runs in results_data.items():
        is_tabpfn = "tabpfn" in model_name.lower()
        
        for run_idx, run in enumerate(runs):
            scenario = run.get('scenario')
            context_size = run.get('context_size')
            fold = run.get('fold')
         
            # Extract times based on model type
            try:
                if is_tabpfn:
                    f_time = run['mem_time_stats']['fit']['time_s']
                    p_time = run['mem_time_stats']['predict']['time_s']
                else:
                    f_time = run.get('fit_time')
                    p_time = run.get('predict_time')
            except KeyError:
                f_time, p_time = None, None

            # Handle missing/failed runs
            if f_time is None or p_time is None:
                warnings.warn(f"Missing time data for {model_name}, scenario '{scenario}', fold {fold}. Skipping run.")
                continue

            # Extract tradeoff data
            try:
                values = run['instance_summary']['NLLH']
                score = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                # N = N_test_instances[(scenario, fold)]
                N = len(run['instance_summary']['NLLH'])
                amortized_time = (f_time + p_time) / N
            except (KeyError, TypeError):
                warnings.warn(f"Missing score/N_test_instances data for {model_name}, scenario '{scenario}', fold {fold}. Skipping tradeoff.")
                score, amortized_time = None, None

            # Route to appropriate "metrics" for plotting
            if time_of is None or time_of == 'fit':
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": "fit", "value": f_time
                })
            if time_of is None or time_of == 'predict':
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": "predict", "value": p_time
                })
            if score is not None and amortized_time is not None:
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": "tradeoff", "amortized_time": amortized_time, "score": score
                })

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No valid time data found to plot.")
    
    # --- FILTERING ---
    all_available_scenarios = sorted(df['scenario'].unique())

    if plot_scenario is not None:
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    # Determine columns for plot grid
    metrics = ['fit', 'predict'] if time_of is None else [time_of]
    metrics.append('tradeoff') # Always add tradeoff column
    context_sizes = sorted(df['context_size'].dropna().unique())

    # 3. Plotting Setup
    fig, axes = plt.subplots(len(scenarios), len(metrics), 
                             figsize=(6 * len(metrics), 5 * len(scenarios)), 
                             squeeze=False)

    model_handles = {}
    special_handles = {}
    has_out_of_bounds = False

    for row_idx, scenario in enumerate(scenarios):
        scenario_max_context = df[df['scenario'] == scenario]['context_size'].max()
        
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            
            for model_name, (color, hatch) in visual_config.items():
                subset = df[(df['model'] == model_name) & 
                            (df['scenario'] == scenario) & 
                            (df['metric'] == metric)]
                
                if subset.empty:
                    continue

                if metric != 'tradeoff':
                    # Standard time logic
                    fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                    final_stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std']).reset_index()
                    final_stats = final_stats.sort_values('context_size')
                    
                    x = final_stats['context_size']
                    y_mean = final_stats['mean']
                    y_std = final_stats['std']
                    current_max_context = x.max()
                else:
                    # Tradeoff Plot logic: Grouping amortized_time & score across folds
                    fold_means = subset.groupby(['context_size', 'fold'])[['amortized_time', 'score']].mean().reset_index()
                    final_stats = fold_means.groupby('context_size')[['amortized_time', 'score']].agg(['mean', 'std']).reset_index()
                    final_stats = final_stats.sort_values('context_size')
                    
                    x = final_stats[('amortized_time', 'mean')]
                    y_mean = final_stats[('score', 'mean')]
                    y_std = final_stats[('score', 'std')]
                    current_max_context = final_stats['context_size'].max()

                line, = ax.plot(x, y_mean, label=model_name, color=color, marker='o')
                line_objs[model_name] = line
                if model_name not in model_handles: 
                    model_handles[model_name] = line
                
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                
                if log_y and metric != 'tradeoff':
                    lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                
                hatch_val = hatch if use_hatching else None
                ax.fill_between(x, lower_bound, upper_bound, 
                                color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                
                # --- EARLY TERMINATION MARKER ---
                if current_max_context < scenario_max_context:
                    last_x = x.iloc[-1]
                    last_y = y_mean.iloc[-1]
                    
                    if log_x:
                        # Handles offset correctly whether purely log or symlog
                        x_offset = last_x * 1.15 if last_x > 0 else last_x + 1e-5
                    else:
                        if metric != 'tradeoff':
                            x_range = scenario_max_context - context_sizes[0]
                            x_offset = last_x + (x_range * 0.04) 
                        else:
                            x_range = df[df['metric']=='tradeoff']['amortized_time'].max()
                            x_offset = last_x + (x_range * 0.04)
                     
                    sc_timeout = ax.scatter(x_offset, last_y, color=color, marker='x', s=60, 
                               linewidth=1.5, zorder=6)
                    
                    if 'Timeout' not in special_handles: 
                        special_handles['Timeout'] = sc_timeout

                # Record best values (lowest score or time)
                current_min = y_mean.min()
                if current_min < abs_best_val:
                    abs_best_val = current_min
                    abs_best_model = model_name
                    best_x = x[y_mean.idxmin()]
                    abs_best_coords = (best_x, current_min)

                if zoom_on_model == model_name:
                    lower_bound_zoom = (y_mean - y_std).min()
                    upper_bound_zoom = (y_mean + y_std).max()
                    zoom_bounds["min"] = min(zoom_bounds["min"], lower_bound_zoom)
                    zoom_bounds["max"] = max(zoom_bounds["max"], upper_bound_zoom)

            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                sc_best = ax.scatter(bx, by, color='gold', marker='*', s=150, 
                                     edgecolor='black', linewidth=0.5, zorder=5)
                
                if 'Best Model' not in special_handles: 
                    special_handles['Best Model'] = sc_best

            # X-Axis configuration
            if log_x:
                if metric == 'tradeoff':
                    # symlog handles values strictly close to 0 gracefully by linearizing them
                    ax.set_xscale('symlog', linthresh=1e-5)
                else:
                    ax.set_xscale('log')
                    ax.set_xticks(context_sizes) 
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Y-Axis configuration
            if log_y and metric != 'tradeoff':
                ax.set_yscale('log')

            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and metric != 'tradeoff' and low <= 0:
                    if zoom_bounds["min"] > 0:
                        low = zoom_bounds["min"] * 0.9
                    else:
                        low = 1e-9
                ax.set_ylim(low, high)

            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                out_of_bounds_labels = []
                for model_name, line in line_objs.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                    if subset.empty: continue
                    
                    if metric != 'tradeoff':
                        fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                        stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std'])
                    else:
                        fold_means = subset.groupby(['context_size', 'fold'])['score'].mean().reset_index()
                        stats = fold_means.groupby('context_size')['score'].agg(['mean', 'std'])
                    
                    if (stats['mean'] - stats['std']).min() > current_ylim_max:
                        out_of_bounds_labels.append(f"{model_name} (≫)")
                        
                if out_of_bounds_labels:
                    has_out_of_bounds = True
                    ax.text(0.95, 0.95, '\n'.join(out_of_bounds_labels), transform=ax.transAxes, 
                            ha='right', va='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

            # Display names & Titles
            if metric == 'fit':
                metric_display_name = "Fit Time (s)"
                ax_title = metric_display_name
            elif metric == 'predict':
                metric_display_name = "Predict Time (s)"
                ax_title = metric_display_name
            else:
                metric_display_name = "Score (NLLH)"
                ax_title = "Tradeoff (Score vs Time)"

            if row_idx == 0:
                ax.set_title(ax_title)
            
            if col_idx == 0:
                ax.set_ylabel(f"{scenario}\n\n{metric_display_name}")
            else:
                ax.set_ylabel(metric_display_name)
            
            # Bottom labels
            if metric == 'tradeoff':
                ax.set_xlabel("Amortized Time (s/pred)")
            else:
                ax.set_xlabel("Context Size")

            ax.grid(True)

    # --- NEW FUNCTIONALITY: Global Horizontal Legend ---
    # Handle the Out-of-Bounds special legend entry
    if has_out_of_bounds:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='black', linewidths=0.5, s=100)

    # Merge handles, ensuring models are listed first
    final_handles = list(model_handles.values()) + list(special_handles.values())
    final_labels = list(model_handles.keys()) + list(special_handles.keys())

    ncols = len(final_handles)
    
    # Calculate dynamic spacing to keep absolute gap consistent regardless of row count
    fig_height = 5 * len(scenarios)
    if plot_title:
        top_space = 1.0
        legend_y = 1.0 - (0.5 / fig_height)
        title_y = 1.0 - (0.15 / fig_height)
        fig.suptitle(plot_title, y=title_y)
    else:
        top_space = 0.7
        legend_y = 1.0 - (0.15 / fig_height)
        
    rect_top = 1.0 - (top_space / fig_height)

    fig.legend(final_handles, final_labels, loc='upper center', 
               bbox_to_anchor=(0.5, legend_y), ncol=ncols)

    plt.tight_layout(rect=[0, 0, 1, rect_top])
        
    return fig

def plot_main_results(results_data, visual_config, log_x=False, log_y=False, 
                    zoom_on_model=None, plot_title=None, mark_best=False, 
                    plot_scenario=None, plot_metric=None, legends_vertical=False, grid_layout=None, use_hatching=False):
    """
    Plots ML experiment results with filtering for specific scenarios and metrics.
    
    Args:
        results_data: Dict of {model_name: list_of_runs}
        visual_config: Dict of {model_name: (color, hatch)}
        log_x, log_y: Booleans for axis scaling
        zoom_on_model: Model name to base y-axis limits on
        mark_best: Boolean to highlight the best point with a golden star
        plot_scenario: String, if provided, only plot this specific scenario
        plot_metric: String, if provided, only plot this specific metric
        legends_vertical: Boolean to stack global legend items vertically
    """
    # 1. Validation
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []
    metrics_list = []

    for model_name, runs in results_data.items():
        for run_idx, run in enumerate(runs):
            scenario = run['scenario']
            context_size = run['context_size']
            fold = run['fold']
            summary = run['instance_summary']
            
            for metric_name, values in summary.items():
                if metric_name != "RMSE" and metric_name not in metrics_list:
                    metrics_list.append(metric_name)
                
                val = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": metric_name, "value": val
                })

    df = pd.DataFrame(all_records)
    
    # --- NEW FUNCTIONALITY: FILTERING ---
    all_available_scenarios = sorted(df['scenario'].unique())
    all_available_metrics = metrics_list # maintaining original order of discovery

    # Handle Scenario filtering
    if plot_scenario is not None:
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    # Handle Metric filtering
    if plot_metric is not None:
        if isinstance(plot_metric, str):
            if plot_metric not in all_available_metrics:
                raise ValueError(f"Metric '{plot_metric}' not found in data. Available: {all_available_metrics}")
            else:
                metrics = [plot_metric]
        elif isinstance(plot_metric, list):
            missing = set(plot_metric) - set(all_available_metrics)
            if missing:
                raise ValueError(f"Metrics {missing} not found in data. Available: {all_available_metrics}")
            else:
                metrics = plot_metric
    else:
        metrics = all_available_metrics
    # ------------------------------------

    context_sizes = sorted(df['context_size'].unique())

    # 3. Plotting Setup
    n_scenarios, n_metrics = len(scenarios), len(metrics)
    total_plots = n_scenarios * n_metrics
    use_custom_grid = grid_layout is not None and (n_scenarios == 1 or n_metrics == 1)

    if use_custom_grid:
        g_rows, g_cols = grid_layout
        if g_rows == -1: g_rows = int(np.ceil(total_plots / g_cols))
        if g_cols == -1: g_cols = int(np.ceil(total_plots / g_rows))
    else:
        g_rows, g_cols = n_scenarios, n_metrics

    fig, axes = plt.subplots(g_rows, g_cols, 
                             figsize=(6 * g_cols, 5 * g_rows), 
                             squeeze=False)
    
                             
    model_handles = {}
    special_handles = {}
    has_out_of_bounds = False

    for row_idx, scenario in enumerate(scenarios):
        scenario_max_context = df[df['scenario'] == scenario]['context_size'].max()
        
        for col_idx, metric in enumerate(metrics):
            if use_custom_grid:
                plot_idx = row_idx * n_metrics + col_idx
                ax_row, ax_col = plot_idx // g_cols, plot_idx % g_cols
                ax = axes[ax_row][ax_col]
            else:
                ax_row, ax_col = row_idx, col_idx
                ax = axes[ax_row][ax_col]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            
            added_early_term_legend = False 
            
            for model_name, (color, hatch) in visual_config.items():
                subset = df[(df['model'] == model_name) & 
                            (df['scenario'] == scenario) & 
                            (df['metric'] == metric)]
                
                if subset.empty:
                    continue

                fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                final_stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std']).reset_index()
                final_stats = final_stats.sort_values('context_size')
                
                x = final_stats['context_size']
                y_mean = final_stats['mean']
                y_std = final_stats['std']
                
                line, = ax.plot(x, y_mean, label=model_name, color=color, marker='o')
                line_objs[model_name] = line
                if model_name not in model_handles: 
                    model_handles[model_name] = line
                
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                
                if log_y and metric != 'NLLH':
                    lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                
                hatch_val = hatch if use_hatching else None
                ax.fill_between(x, lower_bound, upper_bound, 
                                color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                
                # --- EARLY TERMINATION MARKER ---
                current_max_context = x.max()
                if current_max_context < scenario_max_context:
                    last_x = final_stats['context_size'].iloc[-1]
                    last_y = final_stats['mean'].iloc[-1]
                    
                    if log_x:
                        x_offset = last_x * 1.15  
                    else:
                        x_range = scenario_max_context - context_sizes[0]
                        x_offset = last_x + (x_range * 0.04) 
                        
                    label = 'Timeout' if not added_early_term_legend else '_nolegend_'
                    sc = ax.scatter(x_offset, last_y, color=color, marker='x', s=60, 
                               linewidth=1.5, zorder=6, label=label)
                    added_early_term_legend = True
                    if 'Timeout' not in special_handles: 
                        special_handles['Timeout'] = sc
                # ---------------------------------------------------

                current_min = y_mean.min()
                if current_min < abs_best_val:
                    abs_best_val = current_min
                    abs_best_model = model_name
                    best_x = x[y_mean.idxmin()]
                    abs_best_coords = (best_x, current_min)

                if zoom_on_model == model_name:
                    lower_bound = (y_mean - y_std).min()
                    upper_bound = (y_mean + y_std).max()
                    zoom_bounds["min"] = min(zoom_bounds["min"], lower_bound)
                    zoom_bounds["max"] = max(zoom_bounds["max"], upper_bound)


            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                sc_best = ax.scatter(bx, by, color='gold', marker='*', s=150, 
                           edgecolor='black', linewidth=0.5, zorder=5, label='Best Model')
                if 'Best Model' not in special_handles: 
                    special_handles['Best Model'] = sc_best

            if log_x:
                ax.set_xscale('log')
                ax.set_xticks(context_sizes) 
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            if log_y and metric != 'NLLH':
                ax.set_yscale('log')

            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and metric != 'NLLH' and low <= 0:
                    if zoom_bounds["min"] > 0:
                        low = zoom_bounds["min"] * 0.9
                    else:
                        low = 1e-9
                
                ax.set_ylim(low, high)

            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                out_of_bounds_labels = []
                for model_name, line in line_objs.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == metric)]
                    if subset.empty: continue
                    
                    fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                    stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std'])
                    
                    if (stats['mean'] - stats['std']).min() > current_ylim_max:
                        out_of_bounds_labels.append(f"{model_name} (≫)")
                        
                if out_of_bounds_labels:
                    has_out_of_bounds = True
                    ax.text(0.95, 0.95, '\n'.join(out_of_bounds_labels), transform=ax.transAxes, 
                            ha='right', va='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

            # --- NEW: Layout ---
            if len(scenarios) == 1 and len(metrics) == 1:
                # Case 1: Specific Scenario & Specific Metric -> Title becomes Scenario
                ax.set_title(scenario)
                ax.set_ylabel(metric)
            elif use_custom_grid:
                if len(metrics) == 1:
                    ax.set_title(scenario)
                    if ax_col == 0: ax.set_ylabel(metric)
                else:
                    ax.set_title(metric)
                    if ax_col == 0: ax.set_ylabel(scenario)
            else:
                if row_idx == 0: ax.set_title(f"Metric: {metric}")
                ax.set_ylabel(f"{scenario}\n\n{metric}" if col_idx == 0 else metric)
            
            ax.set_xlabel("Context Size")
            ax.grid(True)

    # Handle the Out-of-Bounds special legend entry
    # Handle the Out-of-Bounds special legend entry
    if has_out_of_bounds:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='black', linewidths=0.5, s=100)

    # Merge handles, ensuring models are listed first
    final_handles = list(model_handles.values()) + list(special_handles.values())
    final_labels = list(model_handles.keys()) + list(special_handles.keys())

    ncols = 1 if legends_vertical else len(final_handles)
    
    # --- NEW: Check if there's only one plot ---
    if total_plots == 1:
        # Case: Single plot -> Put legend inside the plot
        single_ax = axes[0][0]
        single_ax.legend(final_handles, final_labels, loc='best', ncol=ncols)
        
        if plot_title:
            fig.suptitle(plot_title)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Slight margin for title
        else:
            plt.tight_layout()
            
    else:
        # Case: Multiple plots -> Keep global legend on top and handle orphan centering
        fig_height = 5 * g_rows
        
        if plot_title:
            top_space = 1.6 if legends_vertical else 1.2
            legend_y = 1.0 - (0.5 / fig_height)
            title_y = 1.0 - (0.15 / fig_height)
            fig.suptitle(plot_title, y=title_y)
        else:
            top_space = 1.3 if legends_vertical else 1.1
            legend_y = 1.0 - (0.15 / fig_height)
            
        rect_top = 1.0 - (top_space / fig_height)

        fig.legend(final_handles, final_labels, loc='upper center', 
              bbox_to_anchor=(0.5, legend_y), ncol=ncols)

        plt.tight_layout(rect=[0, 0, 1, rect_top])
        
        # Center orphan subplots in the last row
        empty_slots = (g_rows * g_cols) - total_plots
        if 0 < empty_slots < g_cols and g_cols > 1:
            pos_0 = axes[0][0].get_position()
            pos_1 = axes[0][1].get_position()
            step = pos_1.x0 - pos_0.x0
            shift = (empty_slots * step) / 2.0
            
            cols_in_last_row = g_cols - empty_slots
            for col in range(cols_in_last_row):
                ax_to_move = axes[g_rows - 1][col]
                pos = ax_to_move.get_position()
                pos.x0 += shift
                pos.x1 += shift
                ax_to_move.set_position(pos)
                
    # Delete any unused subplots completely
    for i in range(total_plots, g_rows * g_cols):
        fig.delaxes(axes[i // g_cols][i % g_cols])
        
    return fig

def plot_vram_results(results_data, visual_config, vram_of=None, log_x=False, log_y=False, 
                      zoom_on_model=None, plot_title=None, mark_best=False, 
                      plot_scenario=None, merge_plots=False, use_hatching=False):
    """
    Plots TabPFN VRAM usage (fit and/or predict) in Gigabytes.
    
    Args:
        results_data: Dict of {model_name: list_of_runs} (Assuming only TabPFN data)
        visual_config: Dict of {model_name: (color, hatch)}
        vram_of: String ('fit', 'predict') or None (plots both side-by-side)
        log_x, log_y: Booleans for axis scaling
        zoom_on_model: Model name to base y-axis limits on
        mark_best: Boolean to highlight the lowest point with a golden star
        plot_scenario: String or list, if provided, only plot specific scenarios
        merge_plots: Boolean, if True, plots fit and predict on the same axis in a grid format
    """
    # 1. Validation
    if vram_of not in [None, 'fit', 'predict']:
        raise ValueError("vram_of must be None, 'fit', or 'predict'")
        
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")

    # 2. Flatten and Pre-process Data
    all_records = []

    for model_name, runs in results_data.items():
        for run_idx, run in enumerate(runs):
            scenario = run.get('scenario')
            context_size = run.get('context_size')
            fold = run.get('fold')
            
            # --- EXTRACT VRAM (TABPFN SPECIFIC) ---
            try:
                # Extract and convert from MB to GB
                f_vram = run['mem_time_stats']['fit']['spike_mb'] / 1024.0
                p_vram = run['mem_time_stats']['predict']['spike_mb'] / 1024.0
            except KeyError:
                f_vram, p_vram = None, None

            # Handle missing/failed runs
            if f_vram is None or p_vram is None:
                warnings.warn(f"Missing VRAM data for {model_name}, scenario '{scenario}', fold {fold}. Skipping run.")
                continue

            # Route to appropriate "metrics" for plotting
            if vram_of is None or vram_of == 'fit':
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": "fit", "value": f_vram
                })
            if vram_of is None or vram_of == 'predict':
                all_records.append({
                    "model": model_name, "scenario": scenario, 
                    "context_size": context_size, "fold": fold, 
                    "metric": "predict", "value": p_vram
                })

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No valid VRAM data found to plot.")
    
    # --- FILTERING ---
    all_available_scenarios = sorted(df['scenario'].unique())

    if plot_scenario is not None:
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    metrics = ['fit', 'predict'] if vram_of is None else [vram_of]
    context_sizes = sorted(df['context_size'].unique())

    # 3. Plotting Setup (Grid Logic)
    # CHANGED: Dynamically constrain columns to actual number of scenarios if less than 3
    n_cols = min(3, len(scenarios)) if merge_plots else len(metrics)
    n_rows = int(np.ceil(len(scenarios) / 3)) if merge_plots else len(scenarios)
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=(6 * n_cols, 5 * n_rows), 
                             squeeze=False)

    if merge_plots:
        for ax in axes.flat:
            ax.set_visible(False)

    global_handles = {} 
    merge_colors = {'fit': 'tab:blue', 'predict': 'tab:orange'}

    for idx, scenario in enumerate(scenarios):
        scenario_max_context = df[df['scenario'] == scenario]['context_size'].max()
        
        loop_cols = [0] if merge_plots else range(n_cols)
        
        for loop_c in loop_cols:
            if merge_plots:
                row_idx = idx // 3
                # CHANGED: Safeguard center logic so it only triggers when there is a real 3-column grid
                if row_idx == n_rows - 1 and len(scenarios) % 3 == 1 and n_cols == 3:
                    col_idx = 1
                else:
                    col_idx = idx % 3
                metrics_to_plot = metrics
            else:
                row_idx = idx
                col_idx = loop_c
                metrics_to_plot = [metrics[col_idx]]

            ax = axes[row_idx][col_idx]
            
            if merge_plots:
                ax.set_visible(True)
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            
            for metric in metrics_to_plot:
                for model_name, (color, hatch) in visual_config.items():
                    subset = df[(df['model'] == model_name) & 
                                (df['scenario'] == scenario) & 
                                (df['metric'] == metric)]
                    
                    if subset.empty:
                        continue

                    fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                    final_stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std']).reset_index()
                    final_stats = final_stats.sort_values('context_size')
                    
                    x = final_stats['context_size']
                    y_mean = final_stats['mean']
                    y_std = final_stats['std']
                    
                    # Override labels/colors if merging plots
                    plot_color = merge_colors[metric] if merge_plots else color
                    plot_hatch = hatch  # Store original hatch
                    final_label = f"Fit VRAM (GB)" if metric == 'fit' else f"Predict VRAM (GB)"
                    if not merge_plots:
                        final_label = model_name

                    line, = ax.plot(x, y_mean, label=final_label, color=plot_color, marker='o')
                    line_objs[final_label] = line
                    
                    if final_label not in global_handles:
                        global_handles[final_label] = line
                    
                    lower_bound = y_mean - y_std
                    upper_bound = y_mean + y_std
                    
                    if log_y:
                        lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                    
                    hatch_val = plot_hatch if use_hatching else None
                    ax.fill_between(x, lower_bound, upper_bound, 
                                    color=plot_color, alpha=0.2, 
                                    hatch=hatch_val, edgecolor='none')
                    
                    # --- EARLY TERMINATION MARKER ---
                    current_max_context = x.max()
                    if current_max_context < scenario_max_context:
                        last_x = final_stats['context_size'].iloc[-1]
                        last_y = final_stats['mean'].iloc[-1]
                        
                        if log_x:
                            x_offset = last_x * 1.15 
                        else:
                            x_range = scenario_max_context - context_sizes[0]
                            x_offset = last_x + (x_range * 0.04) 
                         
                        oom_sc = ax.scatter(x_offset, last_y, color=plot_color, marker='x', s=60, 
                                            linewidth=1.5, zorder=6)
                        
                        if 'OOM' not in global_handles:
                            global_handles['OOM'] = oom_sc

                    current_min = y_mean.min()
                    if current_min < abs_best_val:
                        abs_best_val = current_min
                        abs_best_model = final_label
                        best_x = x[y_mean.idxmin()]
                        abs_best_coords = (best_x, current_min)

                    if zoom_on_model == model_name:
                        lower_bound_zoom = (y_mean - y_std).min()
                        upper_bound_zoom = (y_mean + y_std).max()
                        zoom_bounds["min"] = min(zoom_bounds["min"], lower_bound_zoom)
                        zoom_bounds["max"] = max(zoom_bounds["max"], upper_bound_zoom)

            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                best_sc = ax.scatter(bx, by, color='gold', marker='*', s=150, 
                                     edgecolor='black', linewidth=0.5, zorder=5)
                if 'Best' not in global_handles:
                    global_handles['Best'] = best_sc
                
                original_label = line_objs[abs_best_model].get_label()
                line_objs[abs_best_model].set_label(f"{original_label} <-- Best")

            if log_x:
                ax.set_xscale('log')
                ax.set_xticks(context_sizes) 
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            if log_y:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))

            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and low <= 0:
                    if zoom_bounds["min"] > 0:
                        low = zoom_bounds["min"] * 0.9
                    else:
                        low = 1e-9
                ax.set_ylim(low, high)

            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                for final_label, line in line_objs.items():
                    # Reverse map final_label to metric and model to fetch correct subset bounds
                    if merge_plots:
                        metric_for_subset = 'fit' if 'Fit' in final_label else 'predict'
                        model_for_subset = list(visual_config.keys())[0] # Assumes single model
                    else:
                        metric_for_subset = metrics_to_plot[0]
                        model_for_subset = final_label
                        
                    subset = df[(df['model'] == model_for_subset) & (df['scenario'] == scenario) & (df['metric'] == metric_for_subset)]
                    if subset.empty: continue
                    
                    fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                    stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std'])
                    
                    if (stats['mean'] - stats['std']).min() > current_ylim_max:
                        current_label = line.get_label()
                        if " <-- Best" not in current_label:
                            line.set_label(f"{current_label} (≫)") 

            # Titles and Y-Labels dynamically adjust based on merge state
            if merge_plots:
                ax.set_title(scenario)
                ax.set_ylabel("VRAM (GB)")
            else:
                metric_display_name = "Fit VRAM (GB)" if metrics_to_plot[0] == 'fit' else "Predict VRAM (GB)"
                if row_idx == 0:
                    ax.set_title(metric_display_name)
                if col_idx == 0:
                    ax.set_ylabel(f"{scenario}\n{metric_display_name}")
                else:
                    ax.set_ylabel(metric_display_name)
            
            ax.set_xlabel("Context Size")
            ax.grid(True)

    # --- Global Legend and Layout Finalization ---
    handles = list(global_handles.values())
    labels = list(global_handles.keys())

    fig_height = 5 * n_rows
    if plot_title:
        fig.suptitle(plot_title, y=1.0 - (0.15 / fig_height))
        legend_y = 1.0 - (0.5 / fig_height)
        rect_top = 1.0 - (1.0 / fig_height) # Leave space for title + legend
    else:
        legend_y = 1.0 - (0.15 / fig_height)
        rect_top = 1.0 - (0.7 / fig_height) # Leave space for legend only

    # Render single global legend horizontally with border
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, legend_y), 
               ncol=len(labels))
    
    plt.tight_layout(rect=[0, 0, 1, rect_top])
        
    return fig

def plot_combined_time_vram_results(results_data, visual_config, log_x=False, log_y=False, 
                                    zoom_on_model=None, plot_title=None, mark_best=False, 
                                    plot_scenario=None, use_hatching=False):
    """
    Plots ML experiment wall-clock time results, tradeoff curves, and VRAM usage.
    Outputs a grid where each row is a scenario and columns are:
    [Fit Time, Predict Time, Tradeoff, VRAM]
    """
    # 1. Validation
    if set(results_data.keys()) != set(visual_config.keys()):
        raise KeyError("The keys in results_data and visual_config must match exactly.")
    
    if zoom_on_model is not None and zoom_on_model not in results_data.keys():
        raise ValueError(f"zoom_on_model '{zoom_on_model}' not found in results_data keys.")

    # 2. Flatten and Pre-process Data
    all_records = []

    for model_name, runs in results_data.items():
        is_tabpfn = "tabpfn" in model_name.lower()
        
        for run_idx, run in enumerate(runs):
            scenario = run.get('scenario')
            context_size = run.get('context_size')
            fold = run.get('fold')
         
            # --- PATCH 1: Added TypeError to handle explicit None values in nested dicts ---
            # Extract times
            try:
                if is_tabpfn:
                    f_time = run['mem_time_stats']['fit']['time_s']
                    p_time = run['mem_time_stats']['predict']['time_s']
                else:
                    f_time = run.get('fit_time')
                    p_time = run.get('predict_time')
            except (KeyError, TypeError):
                f_time, p_time = None, None

            # Extract VRAM (TabPFN Specific)
            try:
                if is_tabpfn:
                    f_vram = run['mem_time_stats']['fit']['spike_mb'] / 1024.0
                    p_vram = run['mem_time_stats']['predict']['spike_mb'] / 1024.0
                    total_vram = f_vram + p_vram  # <-- NEW: Calculate total
                else:
                    f_vram, p_vram, total_vram = None, None, None
            except (KeyError, TypeError):
                f_vram, p_vram, total_vram = None, None, None

            # Handle warnings
            if f_time is None or p_time is None:
                warnings.warn(f"Missing time data for {model_name}, scenario '{scenario}', fold {fold}.")
            if (f_vram is None or p_vram is None) and is_tabpfn:
                warnings.warn(f"Missing VRAM data for {model_name}, scenario '{scenario}', fold {fold}.")

            # --- PATCH 2: Added ZeroDivisionError and TypeError safeguards for Tradeoff ---
            try:
                values = run['instance_summary']['NLLH']
                score = values.numpy().mean() if hasattr(values, 'numpy') else np.mean(values)
                N = len(run['instance_summary']['NLLH'])
                if N == 0: raise ZeroDivisionError
                amortized_time = (f_time + p_time) / N if (f_time is not None and p_time is not None) else None
            except (KeyError, TypeError, ZeroDivisionError):
                score, amortized_time = None, None

            # Route to appropriate metrics
            if f_time is not None:
                all_records.append({"model": model_name, "scenario": scenario, "context_size": context_size, "fold": fold, "metric": "fit_time", "value": f_time})
            if p_time is not None:
                all_records.append({"model": model_name, "scenario": scenario, "context_size": context_size, "fold": fold, "metric": "predict_time", "value": p_time})
            if score is not None and amortized_time is not None:
                all_records.append({"model": model_name, "scenario": scenario, "context_size": context_size, "fold": fold, "metric": "tradeoff", "amortized_time": amortized_time, "score": score})
            if total_vram is not None:
                # <-- NEW: Only append the combined total VRAM
                all_records.append({"model": model_name, "scenario": scenario, "context_size": context_size, "fold": fold, "metric": "total_vram", "value": total_vram})

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No valid data found to plot.")
    
    # --- FILTERING ---
    all_available_scenarios = sorted(df['scenario'].unique())

    if plot_scenario is not None:
        requested_scenarios = [plot_scenario] if isinstance(plot_scenario, str) else plot_scenario
        missing = set(requested_scenarios) - set(all_available_scenarios)
        if missing:
            raise ValueError(f"Scenarios {missing} not found in data. Available: {all_available_scenarios}")
        scenarios = requested_scenarios
    else:
        scenarios = all_available_scenarios

    metrics_cols = ['fit_time', 'predict_time', 'tradeoff', 'vram']
    context_sizes = sorted(df['context_size'].dropna().unique())

    # 3. Plotting Setup
    fig, axes = plt.subplots(len(scenarios), len(metrics_cols), 
                             figsize=(6 * len(metrics_cols), 5 * len(scenarios)), 
                             squeeze=False)

    global_handles = {}
    special_handles = {}
    has_out_of_bounds = False
    

    for row_idx, scenario in enumerate(scenarios):
        scenario_max_context = df[df['scenario'] == scenario]['context_size'].max()
        
        for col_idx, col_metric in enumerate(metrics_cols):
            ax = axes[row_idx][col_idx]
            
            zoom_bounds = {"min": float('inf'), "max": float('-inf')}
            abs_best_val = float('inf')
            abs_best_model = None
            abs_best_coords = (None, None)
            line_objs = {}
            
            # -------------------------------------------------------------
            # LOGIC FOR TIME AND TRADEOFF PLOTS (Col 0, 1, 2)
            # -------------------------------------------------------------
            if col_metric != 'vram':
                for model_name, (color, hatch) in visual_config.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == col_metric)]
                    if subset.empty: continue

                    if col_metric != 'tradeoff':
                        fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                        final_stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std']).reset_index().sort_values('context_size')
                        
                        x = final_stats['context_size']
                        y_mean = final_stats['mean']
                        # --- PATCH 3: fillna(0) to prevent matplotlib crashing on 0-variance fill_betweens ---
                        y_std = final_stats['std'].fillna(0)
                        current_max_context = x.max()
                    else:
                        fold_means = subset.groupby(['context_size', 'fold'])[['amortized_time', 'score']].mean().reset_index()
                        final_stats = fold_means.groupby('context_size')[['amortized_time', 'score']].agg(['mean', 'std']).reset_index().sort_values('context_size')
                        
                        x = final_stats[('amortized_time', 'mean')]
                        y_mean = final_stats[('score', 'mean')]
                        y_std = final_stats[('score', 'std')].fillna(0)
                        current_max_context = final_stats['context_size'].max()

                    line, = ax.plot(x, y_mean, label=model_name, color=color, marker='o')
                    line_objs[model_name] = line
                    if model_name not in global_handles: 
                        global_handles[model_name] = line
                    
                    lower_bound = y_mean - y_std
                    upper_bound = y_mean + y_std
                    
                    if log_y and col_metric != 'tradeoff':
                        lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                    
                    hatch_val = hatch if use_hatching else None
                    ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                    
                    # EARLY TERMINATION MARKER
                    if current_max_context < scenario_max_context:
                        last_x = x.iloc[-1]
                        last_y = y_mean.iloc[-1]
                        
                        if log_x:
                            x_offset = last_x * 1.15 if last_x > 0 else last_x + 1e-5
                        else:
                            if col_metric != 'tradeoff':
                                x_range = scenario_max_context - context_sizes[0]
                                x_offset = last_x + (x_range * 0.04) 
                            else:
                                x_range = df[df['metric']=='tradeoff']['amortized_time'].max()
                                x_offset = last_x + (x_range * 0.04)
                           
                        sc_timeout = ax.scatter(x_offset, last_y, color=color, marker='x', s=60, linewidth=1.5, zorder=6)
                        if 'Timeout' not in special_handles: 
                            special_handles['Timeout'] = sc_timeout

                    # BEST MODEL MARKER
                    current_min = y_mean.min()
                    if current_min < abs_best_val:
                        abs_best_val = current_min
                        abs_best_model = model_name
                        abs_best_coords = (x[y_mean.idxmin()], current_min)

                    if zoom_on_model == model_name:
                        zoom_bounds["min"] = min(zoom_bounds["min"], (y_mean - y_std).min())
                        zoom_bounds["max"] = max(zoom_bounds["max"], (y_mean + y_std).max())

            # -------------------------------------------------------------
            # LOGIC FOR VRAM PLOT (Col 3) - Total VRAM logic
            # -------------------------------------------------------------
            else: 
                for model_name, (color, hatch) in visual_config.items():
                    subset = df[(df['model'] == model_name) & (df['scenario'] == scenario) & (df['metric'] == 'total_vram')]
                    if subset.empty: continue
                    
                    fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                    final_stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std']).reset_index().sort_values('context_size')
                    
                    x = final_stats['context_size']
                    y_mean = final_stats['mean']
                    y_std = final_stats['std'].fillna(0)
                    
                    plot_color = color  # <-- NEW: Uses the model's actual dictionary color
                    hatch_val = hatch if use_hatching else None
                    final_label = "TabPFN (Fit + Predict VRAM in GB)"  # <-- NEW: Unified label
                    
                    line, = ax.plot(x, y_mean, label=final_label, color=plot_color, marker='o')
                    line_objs[final_label] = line
                    if final_label not in global_handles:
                        global_handles[final_label] = line
                          
                    lower_bound = y_mean - y_std
                    upper_bound = y_mean + y_std
                    
                    if log_y:
                        lower_bound = np.maximum(lower_bound, y_mean * 0.1)
                            
                    ax.fill_between(x, lower_bound, upper_bound, color=plot_color, alpha=0.2, hatch=hatch_val, edgecolor='none')
                    
                    # OOM EARLY TERMINATION
                    current_max_context = x.max()
                    if current_max_context < scenario_max_context:
                        last_x = final_stats['context_size'].iloc[-1]
                        last_y = final_stats['mean'].iloc[-1]
                        
                        if log_x:
                            x_offset = last_x * 1.15 
                        else:
                            x_range = scenario_max_context - context_sizes[0]
                            x_offset = last_x + (x_range * 0.04) 
                          
                        oom_sc = ax.scatter(x_offset, last_y, color=plot_color, marker='x', s=60, linewidth=1.5, zorder=6)
                        if 'OOM' not in special_handles:
                            special_handles['OOM'] = oom_sc

                    # BEST VRAM MARKER
                    current_min = y_mean.min()
                    if current_min < abs_best_val:
                        abs_best_val = current_min
                        abs_best_model = final_label
                        abs_best_coords = (x[y_mean.idxmin()], current_min)

                    # BOUNDS
                    if zoom_on_model == model_name: 
                        zoom_bounds["min"] = min(zoom_bounds["min"], (y_mean - y_std).min())
                        zoom_bounds["max"] = max(zoom_bounds["max"], (y_mean + y_std).max())
            # -------------------------------------------------------------
            # SHARED PLOT FORMATTING
            # -------------------------------------------------------------
            if mark_best and abs_best_model is not None:
                bx, by = abs_best_coords
                sc_best = ax.scatter(bx, by, color='gold', marker='*', s=150, edgecolor='black', linewidth=0.5, zorder=5)
                
                best_leg_label = 'Best Model' if col_metric != 'vram' else 'Best'
                if best_leg_label not in special_handles: 
                    special_handles[best_leg_label] = sc_best
                
                original_label = line_objs[abs_best_model].get_label()
                line_objs[abs_best_model].set_label(f"{original_label} <-- Best")

            # X-Axis configuration
            if log_x:
                if col_metric == 'tradeoff':
                    ax.set_xscale('symlog', linthresh=1e-5)
                else:
                    ax.set_xscale('log')
                    ax.set_xticks(context_sizes) 
                    
                    if col_metric == 'vram':
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
                    else:
                        ax.xaxis.set_major_formatter(ScalarFormatter())
                        
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Y-Axis configuration
            if log_y and col_metric != 'tradeoff':
                ax.set_yscale('log')
                if col_metric == 'vram':
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:g}"))

            if zoom_on_model is not None:
                margin = (zoom_bounds["max"] - zoom_bounds["min"]) * 0.05
                low, high = zoom_bounds["min"] - margin, zoom_bounds["max"] + margin
                if log_y and col_metric != 'tradeoff' and low <= 0:
                    low = zoom_bounds["min"] * 0.9 if zoom_bounds["min"] > 0 else 1e-9
                ax.set_ylim(low, high)

            if zoom_on_model is not None:
                current_ylim_max = ax.get_ylim()[1]
                out_of_bounds_labels = []
                for lbl, line in line_objs.items():
                    if col_metric != 'vram':
                        subset = df[(df['model'] == lbl) & (df['scenario'] == scenario) & (df['metric'] == col_metric)]
                        if subset.empty: continue
                        
                        if col_metric != 'tradeoff':
                            fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                            stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std'])
                        else:
                            fold_means = subset.groupby(['context_size', 'fold'])['score'].mean().reset_index()
                            stats = fold_means.groupby('context_size')['score'].agg(['mean', 'std'])
                        
                        if (stats['mean'] - stats['std'].fillna(0)).min() > current_ylim_max:
                            out_of_bounds_labels.append(f"{lbl} (≫)")
                            
                    else: # VRAM Subsets logic
                        vram_metric = 'total_vram'
                        # Target the tabpfn model explicitly so we can safely check its bounds against the active ax limits
                        model_to_check = next((m for m in visual_config.keys() if "tabpfn" in m.lower()), list(visual_config.keys())[0])
                        
                        subset = df[(df['model'] == model_to_check) & (df['scenario'] == scenario) & (df['metric'] == vram_metric)]
                        if subset.empty: continue
                        
                        fold_means = subset.groupby(['context_size', 'fold'])['value'].mean().reset_index()
                        stats = fold_means.groupby('context_size')['value'].agg(['mean', 'std'])
                        
                        if (stats['mean'] - stats['std'].fillna(0)).min() > current_ylim_max:
                            if " <-- Best" not in lbl:
                                line.set_label(f"{lbl} (≫)")
                                
                if out_of_bounds_labels and col_metric != 'vram':
                    has_out_of_bounds = True
                    ax.text(0.95, 0.95, '\n'.join(out_of_bounds_labels), transform=ax.transAxes, 
                            ha='right', va='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

            # Titles and Labels mapping
            if col_metric == 'fit_time':
                ax_title, y_label, x_label = "Fit Time (s)", "Fit Time (s)", "Context Size"
            elif col_metric == 'predict_time':
                ax_title, y_label, x_label = "Predict Time (s)", "Predict Time (s)", "Context Size"
            elif col_metric == 'tradeoff':
                ax_title, y_label, x_label = "Tradeoff (Score vs Time)", "Score (NLLH)", "Amortized Time (s/pred)"
            else: # vram
                ax_title, y_label, x_label = "VRAM (GB)", "VRAM (GB)", "Context Size"

            if row_idx == 0:
                ax.set_title(ax_title)
            
            if col_idx == 0:
                ax.set_ylabel(f"{scenario}\n\n{y_label}")
            else:
                ax.set_ylabel(y_label)
            
            ax.set_xlabel(x_label)
            ax.grid(True)

    # --- Global Legend and Layout Finalization ---
    if has_out_of_bounds:
        special_handles['Out-of-Bounds'] = ax.scatter([], [], marker='$\\gg$', facecolors='none', edgecolors='black', linewidths=0.5, s=100)

    final_handles = list(global_handles.values()) + list(special_handles.values())
    final_labels = list(global_handles.keys()) + list(special_handles.keys())
    
    fig_height = 5 * len(scenarios)
    if plot_title:
        top_space = 1.0
        legend_y = 1.0 - (0.5 / fig_height)
        title_y = 1.0 - (0.15 / fig_height)
        fig.suptitle(plot_title, y=title_y)
    else:
        top_space = 0.7
        legend_y = 1.0 - (0.10 / fig_height)
        
    rect_top = 1.0 - (top_space / fig_height)

    fig.legend(final_handles, final_labels, loc='upper center', 
               bbox_to_anchor=(0.5, legend_y), ncol=len(final_labels))

    plt.tight_layout(rect=[0, 0, 1, rect_top])
        
    return fig
# -----

@contextlib.contextmanager
def track_gpu_memory_and_time(device_input):
    is_cuda = False
    try:
        device = torch.device(device_input)
        if device.type == 'cuda' and torch.cuda.is_available():
            is_cuda = True
    except Exception:
        pass

    stats = {"baseline_mb": 0.0, "peak_mb": 0.0, "spike_mb": 0.0, "time_s": 0.0}
    if is_cuda:
        gc.collect() 
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        baseline_mem_bytes = torch.cuda.memory_allocated(device)
    
    start_time = time.perf_counter()
    yield stats
    
    if is_cuda:
        torch.cuda.synchronize(device)
        
    stats["time_s"] = time.perf_counter() - start_time
    if is_cuda:
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        stats["baseline_mb"] = baseline_mem_bytes / (1024 ** 2)
        stats["peak_mb"] = peak_mem_bytes / (1024 ** 2)
        stats["spike_mb"] = (peak_mem_bytes - baseline_mem_bytes) / (1024 ** 2)

def sample_k_per_instance(X, y, ids, k, seed):
    """
    Samples up to k random samples for each unique instance id without replacement.
    
    Args:
        X (np.ndarray): Flattened features of shape (M, D)
        y (np.ndarray): Flattened targets of shape (M, 1)
        ids (np.ndarray): Flattened instance ids of shape (M,)
        k (int): Max number of samples to keep per unique id.
        seed (int): Random seed for reproducibility.
        
    Returns:
        X_sampled, y_sampled, ids_sampled
    """
    rng = np.random.RandomState(seed)
    
    # 1. Generate random noise for every single row
    # This determines the "random selection" when we sort
    noise = rng.rand(len(ids))
    
    # 2. Sort by instance_id (primary) and then by noise (secondary)
    # np.lexsort sorts by the last sequence provided first.
    # This groups all identical IDs together, but shuffles them internally.
    sorted_indices = np.lexsort((noise, ids))
    
    # 3. Reorder the ids to find where each group starts
    sorted_ids = ids[sorted_indices]
    
    # 4. Identify boundaries between different instance IDs
    # mask[i] is True if sorted_ids[i] is the start of a new ID group
    mask = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    
    # 5. Calculate the rank of each element within its own ID group
    # Find the index where each group starts
    start_indices = np.where(mask)[0]
    # For every row, find the index where its group started
    # np.diff calculates the size of each group; np.repeat maps the start_index to all rows in that group
    group_offsets = np.repeat(start_indices, np.diff(np.append(start_indices, len(ids))))
    
    # Rank is: (Current Index in sorted array) - (Index where this ID group started)
    # This creates sequences like: [0, 1, 2, 0, 1, 0, 1, 2, 3...]
    ranks = np.arange(len(ids)) - group_offsets
    
    # 6. Filter: Keep only those with a rank < k
    # This naturally handles cases where an instance has fewer than k samples
    keep_mask = ranks < k
    
    # 7. Map the mask back to the original data indexing
    final_indices = sorted_indices[keep_mask]
    
    return X[final_indices], y[final_indices], ids[final_indices]

def generate_experiment_id(cfg) -> str:
    """
    Generates a unique, human-readable identifier based on the experiment configuration.
    This ID is used to ensure that files are not overwritten when hyperparameters change.
    """
    # We use a list of tuples (label, value) to keep the ID organized and readable
    params = [
        (None, cfg.model_name),
        (None, cfg.scenario),
        (None, str(cfg.fold)),
        ("samp", f"{cfg.num_samples_per_instance}_s{cfg.seed_samples_per_instance}"),
        ("ctx", f"{cfg.context_size}_s{cfg.seed_context_size}"),
        ("drop", f"{cfg.feature_drop_rate}_keep{cfg.n_features_keep}_s{cfg.seed_feature_drop_rate}"),
        ("remove_dups", cfg.remove_duplicates),
        ("scale", cfg.target_scale.value),
        ("oracle", cfg.oracle),
        ("es", cfg.early_stopping),
        ("jitterX", f"{cfg.jitter_x}_val{cfg.jitter_val}"),
        ("randExt", f"{cfg.rand_extend_x}_n{cfg.n_rand_cols}"),
        (None, 'cpu' if cfg.use_cpu else 'gpu')
    ]
    
    parts = [f"{label}{val}" if label else str(val) for label, val in params]
    
    return "_".join(parts)

class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PosixPath' and 'pathlib' in module:
            return pathlib.WindowsPath
        return super().find_class(module, name)

def analyze_optimal_context(
    model_results_path,
    scenarios,
    metric='NLLH',
    output_csv_path=None,
    display_results=True
):
    """
    Analyze optimal context sizes across scenarios and optionally save results.
    
    Args:
        model_results_path: Path to pickle file containing model results
        scenarios: List of scenario names to analyze
        metric: Metric to optimize for (default: 'NLLH')
        output_csv_path: Optional path to save results as CSV
        display_results: Whether to display results while analyzing (default: True)
    
    Returns:
        pd.DataFrame: Results with columns 'scenario', 'metric', 'optimal_context'
    """
    import pickle
    import pandas as pd
    from tabpfn_project.helper.utils import find_optimal_context
    
    with open(model_results_path, "rb") as f:
        model_results = pickle.load(f)
    
    rows = []
    for scenario in scenarios:
        optimal_context, summary_df = find_optimal_context(
            model_results, scenario, metric, display_results=display_results
        )
        rows.append({
            "scenario": scenario,
            "metric": metric,
            "optimal_context": optimal_context,
        })
    
    results_df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Saved CSV to: {output_csv_path}")
    
    return results_df

def find_optimal_context(results_list, scenario, target_metric, alpha=0.05, display_results=True):
    """
    Rigorously determines the optimal context size using the Principle of Parsimony 
    and a one-sided Wilcoxon signed-rank test.
    
    Parameters:
    - results_list: List of dictionaries loaded from the .pkl file
    - scenario: String, the dataset name to filter by
    - target_metric: String, e.g., "CRPS", "NLLH", "Wasserstein", "KS"
    - alpha: Float, the statistical significance threshold (default 0.05)
    
    Returns:
    - optimal_context: The selected context size (integer)
    - summary_df: A pandas DataFrame containing the thesis-ready results table
    """
    
    # ---------------------------------------------------------
    # STEP 1: Data Extraction & Instance Aggregation
    # ---------------------------------------------------------
    records = []
    for run in results_list:
        if run['scenario'] == scenario:
            # Aggregate the 200 instances by taking the mean for the target metric
            run_data = run['instance_summary'][target_metric]
            if hasattr(run_data, 'cpu'):
                run_data = run_data.detach().cpu().numpy()
            run_score = np.mean(run_data)
            records.append({
                'context_size': run['context_size'],
                'fold': run['fold'],
                'seed': run['seed_context_size'],
                'score': run_score
            })
            
    if not records:
        raise ValueError(f"No data found for scenario '{scenario}'. Please check the name.")
        
    df = pd.DataFrame(records)
    
    # ---------------------------------------------------------
    # STEP 2: Seed Aggregation (Handling intra-fold correlation)
    # ---------------------------------------------------------
    # Average the 5 seeds for every (context_size, fold) combination
    fold_df = df.groupby(['context_size', 'fold'])['score'].mean().reset_index()
    
    # ---------------------------------------------------------
    # STEP 3: Identify the Empirical Best (C_best)
    # ---------------------------------------------------------
    # Calculate the grand mean across the 10 independent folds
    summary_df = fold_df.groupby('context_size')['score'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.sort_values('context_size').reset_index(drop=True)
    
    # Find the context_size with the absolute lowest mean (since lower error is better)
    c_best_idx = summary_df['mean'].idxmin()
    c_best = int(summary_df.loc[c_best_idx, 'context_size'])
    
    # Extract the 10 matched fold scores for the empirical best, ensuring order by fold
    scores_best = fold_df[fold_df['context_size'] == c_best].sort_values('fold')['score'].values
    
    # ---------------------------------------------------------
    # STEP 4: Wilcoxon-Parsimony Test
    # ---------------------------------------------------------
    results_log =[]
    optimal_context = None
    
    for _, row in summary_df.iterrows():
        c_test = int(row['context_size'])
        mean_test = row['mean']
        std_test = row['std']
        
        # Extract the 10 matched fold scores for the current candidate
        scores_test = fold_df[fold_df['context_size'] == c_test].sort_values('fold')['score'].values
        
        if c_test == c_best:
            p_value = 1.0  # Comparing to itself
            is_worse = False
            status = "*** Empirical Best ***"
        else:
            # One-sided Wilcoxon test
            # H0: c_test and c_best are symmetric.
            # Ha (greater): scores_test > scores_best (i.e., c_test is strictly WORSE than c_best)
            try:
                # Suppress scipy ties warning temporarily to keep terminal clean
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = wilcoxon(scores_test, scores_best, alternative='greater')
            except ValueError:
                # Triggers if differences are exactly 0 across all 10 folds
                print(f"differences are exactly 0 across all 10 folds")
                p_value = 1.0 
                
            is_worse = p_value < alpha
            status = "Significantly Worse" if is_worse else "Statistical Tie"
            
        # Log the result for the thesis table
        results_log.append({
            'Context Size': c_test,
            f'Mean {target_metric}': mean_test,
            f'Std {target_metric}': std_test,
            'p-value (vs Best)': p_value,
            'Status': status
        })
        
        # Apply Parsimony Rule: 
        # Select the *smallest* context size that is <= c_best and NOT significantly worse.
        # Since we iterate in ascending order, the very first non-worse candidate is our optimal!
        if not is_worse and optimal_context is None and c_test <= c_best:
            optimal_context = c_test

    # ---------------------------------------------------------
    # STEP 5: Formatting and Thesis-Ready Output
    # ---------------------------------------------------------
    log_df = pd.DataFrame(results_log)
    # Update the status string of the chosen optimal context for display purposes
    idx_optimal = log_df.index[log_df['Context Size'] == optimal_context].tolist()[0]
    if optimal_context != c_best:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN (Parsimony)"
    else:
        log_df.at[idx_optimal, 'Status'] += " <-- CHOSEN"
        
    if display_results:        
        # Formatting for terminal beauty
        print(f"\n{'='*75}")
        print(f" Parsimony Analysis | Scenario: {scenario} | Metric: {target_metric}")
        print(f"{'='*75}")
        print(f"Empirical Best Context Size (C_best) : {c_best}")
        print(f"Optimal Context Size Chosen          : {optimal_context}")
        print("-" * 75)
        
        # Pandas display formatting
        formatters = {
            f'Mean {target_metric}': '{:.5f}'.format,
            f'Std {target_metric}': '{:.5f}'.format,
            'p-value (vs Best)': '{:.4f}'.format
        }
        print(log_df.to_string(index=False, formatters=formatters))
        print(f"{'='*75}\n")
    
    return optimal_context, log_df

def dict_to_cpu(d):
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu()
        elif isinstance(v, dict):
            result[k] = dict_to_cpu(v)
        elif isinstance(v, (list, tuple)):
            processed_iterable = [
                vi.detach().cpu() if isinstance(vi, torch.Tensor) 
                else dict_to_cpu(vi) if isinstance(vi, dict) 
                else vi 
                for vi in v
            ]
            result[k] = type(v)(processed_iterable) # keep it a list or tuple
        elif hasattr(v, 'cpu') and callable(v.cpu):
            result[k] = deepcopy(v).cpu()
        else:
            result[k] = v
    return result

def load_pickle(path, access_mode='rb'):
    with open(path, access_mode) as f:
        # Use our custom Unpickler on Windows, otherwise standard pickle
        if platform.system() == 'Windows':
            results_dict = WindowsPathUnpickler(f).load()
        else:
            results_dict = pickle.load(f)
    return results_dict

def subsample_data(*arrays, context_size, seed, with_replacement=True):
    """
    Subsamples the dataset.
    
    Args:
        *arrays: Any number of numpy arrays.
                 All must have the same length in the first dimension.
        context_size: Total number of items to return.
        seed: Random seed for reproducibility.
        
    Returns:
        List of subsampled arrays corresponding to the inputs.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
        
    n_samples = arrays[0].shape[0]
    
    # Assert all arrays have the same number of instances
    assert all(arr.shape[0] == n_samples for arr in arrays), \
        "All arrays must have the same number of instances (shape[0])."
    
    rng = np.random.default_rng(seed)
    
    selected_indices = rng.choice(n_samples, size=context_size, replace=with_replacement)
    return tuple(arr[selected_indices] for arr in arrays)

def append_random_columns(
    *arrays,
    n_random_cols: int = 1, 
    random_state: Optional[Union[int, np.random.Generator]] = None
):
    """
    Augments the feature matrices by appending entirely random continuous columns.
    
    Args:
        X (np.ndarray): The flattened, standardized feature matrix of shape (N, d).
        n_random_cols (int): The number of independent random columns to append.
        random_state (int, Generator, optional): Seed or RNG for reproducibility.
        
    Returns:
        List[np.ndarray, ...]: Augmented feature matrices, each of shape (N, d + n_random_cols).
    """
    N = arrays[0].shape[0]
    D = arrays[0].shape[1]
    assert all(arr.shape[1] == D for arr in arrays), "All input arrays must have the same number of features (shape[1])."

    if n_random_cols < 1:
        return arrays  # No augmentation needed, return original arrays
        
    rng = np.random.default_rng(random_state)
    
    augmented_arrays = tuple(np.hstack((arr, rng.standard_normal(
        size=(arr.shape[0], n_random_cols), 
        dtype=np.float64
    ))) for arr in arrays)
    
    return augmented_arrays

def add_feature_jitter(
    X: np.ndarray, 
    jitter_intensity: float, 
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> np.ndarray:
    """
    Injects numerically stable, zero-mean Gaussian noise into feature matrix X.
    
    Args:
        X (np.ndarray): Input feature matrix of shape (N, d).
        jitter_intensity (float): The relative scaling factor for the noise (alpha).
                                  Default is 1e-3.
        random_state (int, Generator, optional): Seed or RNG for reproducibility.
        
    Returns:
        np.ndarray: A new jittered feature matrix of shape (N, d).
    """
    X_jittered = np.array(X, dtype=np.float64, copy=True)
    
    rng = np.random.default_rng(random_state)
    
    base_std = np.std(X_jittered, axis=0)
        
    noise_stds = jitter_intensity * base_std
    
    noise = rng.normal(loc=0.0, scale=noise_stds, size=X_jittered.shape)
    
    return X_jittered + noise

def subsample_features(
    X_train: np.ndarray, 
    *arrays,
    drop_rate: Optional[float] = None, 
    n_features_keep: Optional[int] = None, 
    seed
) -> Tuple[np.ndarray, ...]:
    """
    Randomly samples a subset of features from the input arrays.
    
    Args:
        X_train: (n_samples, n_features)
        *arrays: Additional arrays with shape (n_samples, n_features)
        drop_rate: Fraction of features to drop (0.0 to 1.0). Ignored if n_features_keep is set.
        n_features_keep: Absolute number of features to keep.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of subsampled arrays, with the same order as input (X_train, *arrays)
    """
    n_features = X_train.shape[1]
    
    if n_features_keep is not None:
        size_features = n_features_keep
    elif drop_rate is not None:
        size_features = int(n_features * (1.0 - drop_rate))
    else:
        # Default: keep everything if no parameters are provided
        size_features = n_features

    # We trigger this if size_features is 0 or if drop_rate was explicitly >= 1.0
    if size_features <= 0 or (drop_rate is not None and drop_rate >= 1.0):
        dummy_X_train = X_train[:, :1] * 0.0
        processed_arrays = [arr[:, :1] * 0.0 for arr in arrays]
        return (dummy_X_train, *processed_arrays)
    
    if size_features == n_features:
        # No subsampling needed, return original arrays
        return (X_train, *arrays)

    if size_features > n_features:
        raise ValueError(f"n_features_keep ({size_features}) cannot be greater than total features ({n_features})")

    assert all(arr.shape[1] == n_features for arr in arrays), (
        "All input arrays must have the same number of features (shape[1])."
    )
    
    rng = np.random.default_rng(seed=seed)
    feature_idx = rng.choice(n_features, size=size_features, replace=False)
    
    # Apply selection to X_train and all auxiliary arrays
    processed_arrays = [arr[:, feature_idx] for arr in arrays]

    return (X_train[:, feature_idx], *processed_arrays)

def subsample_targets_per_instance(y_train, num_samples_per_instance, seed):
    """
    Subsamples a specified number of samples per instance from the training data independently per row.
    
    Args:
        y_train: (n_instances, n_samples) - The training labels.
        num_samples_per_instance: The number of samples to subsample per instance.
        seed_samples_per_instance: The random seed for reproducibility.

    Returns:
        y_train_subsampled: (n_instances, num_samples_per_instance) - The subsampled training labels.
    """
    rng = np.random.default_rng(seed=seed)
    
    # Generate a matrix of random floats with the same shape as y_train
    rand_matrix = rng.random(y_train.shape)
    
    # Argsort each row to get random permutations of indices per row.
    # Take the first `num_samples_per_instance` indices for each row.
    # This mathematically guarantees independent sampling WITHOUT replacement per row.
    subsample_idx = np.argsort(rand_matrix, axis=1)[:, :num_samples_per_instance]
    
    # Create row indices for advanced indexing
    row_idx = np.arange(y_train.shape[0])[:, None]
    
    # Extract the independently sampled runtimes
    y_train_subsampled = y_train[row_idx, subsample_idx]
    
    return y_train_subsampled

def load_tabpfn_preds(cfg, tabpfn_preds_dir):
    exp_id = generate_experiment_id(cfg)
    fname = f"tabpfn_{exp_id}_test_preds.pkl"
    fpath = pathlib.Path(tabpfn_preds_dir) / fname

    with open(fpath, "rb") as f:
        if platform.system() == "Windows":
            from tabpfn_project.helper.utils import WindowsPathUnpickler 
            return WindowsPathUnpickler(f).load()
        return pickle.load(f)

def fetch_save_dict(
    results_dir: pathlib.Path,
    metadata_dir: pathlib.Path,
    model_name: str,
    save_name: str,
    search_key: str | None = None,
    search_value=None,
    scenario: str | None = None,
) -> None:
    """
    Build and save a normalized list of experiment results filtered by model/scenario
    and optional key/value filter, aligned with current metadata produced by scripts/main.py
    and scripts/model_handler.py.
    """
    experiment_results_lst = []

    for fpath in sorted(metadata_dir.glob("*.pkl")):
        results_dict = load_pickle(fpath)

        # Apply filters
        if results_dict.get("model_name") != model_name:
            continue
        if scenario is not None and results_dict.get("scenario") != scenario:
            continue
        if search_key is not None and results_dict.get(search_key) != search_value:
            continue

        result_metrics = results_dict.get("result_metrics") or {}
        model_specific_info = results_dict.get("model_specific_info") or {}

        temp = {
            # Experiment identifiers & config from main.py
            "model_name": results_dict.get("model_name"),
            "save_name": save_name,
            "scenario": results_dict.get("scenario"),
            "fold": results_dict.get("fold"),
            
            # Data preprocessing parameters
            "context_size": results_dict.get("context_size"),
            "seed_context_size": results_dict.get("seed_context_size"),
            "seed_feature_drop_rate": results_dict.get("seed_feature_drop_rate"),
            "seed_samples_per_instance": results_dict.get("seed_samples_per_instance"),
            "feature_drop_rate": results_dict.get("feature_drop_rate"),
            "num_samples_per_instance": results_dict.get("num_samples_per_instance"),
            "n_features_keep": results_dict.get("n_features_keep"),
            "train_size": results_dict.get("train_size"),
            "test_size": results_dict.get("test_size"),
            
            # Feature augmentation
            "jitter_x": results_dict.get("jitter_x"),
            "jitter_val": results_dict.get("jitter_val"),
            "rand_extend_x": results_dict.get("rand_extend_x"),
            "n_rand_cols": results_dict.get("n_rand_cols"),
            
            # Model & evaluation settings
            "target_scale": results_dict.get("target_scale"),
            "use_cpu": results_dict.get("use_cpu"),
            "remove_duplicates": results_dict.get("remove_duplicates"),
            "oracle": results_dict.get("oracle"),
            "do_hpo": results_dict.get("do_hpo"),
            
            # Paths
            "save_dir": results_dict.get("save_dir"),
            
            # Model outputs
            "y_test_preds": results_dict.get("y_test_preds"),
            "instance_summary": result_metrics.get("instance_summary"),
            
            # Model-specific metrics
            "fit_time": model_specific_info.get("fit_time"),
            "predict_time": model_specific_info.get("predict_time"),
            "mem_time_stats": model_specific_info.get("mem_time_stats"),
            "best_epoch": model_specific_info.get("best_epoch"),
            "n_epochs": model_specific_info.get("n_epochs"),
            "y_scaler": model_specific_info.get("y_scaler"),
            "model_config": model_specific_info.get("model_config"),
            "rf_new_default": model_specific_info.get("rf_new_default"),
            "n_samples": model_specific_info.get("n_samples"),
            "n_features": model_specific_info.get("n_features"),
            
            # RF-specific
            "best_hyperparameters": model_specific_info.get("best_hyperparameters"),
            "num_unique_configs": model_specific_info.get("num_unique_configs"),
            "num_finished_trials": model_specific_info.get("num_finished_trials"),
            "num_submitted_trials": model_specific_info.get("num_submitted_trials"),
        }

        experiment_results_lst.append(temp)

    scenario_label = '_' + scenario if scenario is not None else ""
    if search_key is None:
        output_name = f"{save_name}{scenario_label}.pkl"
    else:
        output_name = f"{save_name}_{search_key}_{search_value}{scenario_label}.pkl"

    save_file_path = results_dir / output_name
    with open(save_file_path, "wb") as f:
        pickle.dump(experiment_results_lst, f)

    print(f"Saved to {save_file_path}")
