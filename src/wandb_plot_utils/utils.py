import wandb
import pandas as pd
import matplotlib.style as style
import matplotlib.pyplot as plt
import numpy as np
import plotnine as gg
import seaborn as sns

def setup_style():
    """Updates the theme to preferred default settings"""
    # gg.theme_set(gg.theme_bw(base_size=18))
    # gg.theme_update(figure_size=(6, 4), panel_spacing_x=0.5, panel_spacing_y=0.5)
    # style.use('seaborn')
    style.use('ggplot')
    plt.rc('figure', dpi=256)
    plt.rc('font', family='serif', size=12)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'''
           \usepackage{amsmath,amsfonts}
           \renewcommand{\v}[1]{\boldsymbol{#1}}''')

def find_sweep_variables(df):
  sweep_vars = []
  if df.empty:
      return sweep_vars
      
  for k in df.columns.tolist():
    col = df[k]
    # Handle potentially unhashable types if needed, but original logic was simple
    try:
        if isinstance(col.iloc[0], list):
          col = col.map(tuple)
    except (IndexError, TypeError):
        pass # Handle cases with empty cols or non-list types safely if needed
        
    if (col.nunique() > 1):
      sweep_vars.append(k)
  return sweep_vars

def load_experiment_group_state(group_name, metric_names, entity, project, state="finished", samples=1500):
  """
  Loads experiment data for a given group.
  
  Args:
      group_name: The experiment group name.
      metric_names: List of metrics to fetch history for.
      entity: WANDB entity.
      project: WANDB project.
      state: Filter runs by state (default "finished"). Set to None for all.
  """
  print(f"Entity: {entity}")
  print(f"Group: {group_name}")
  print(f"Project: {project}")
  
  api = wandb.Api()
  all_runs = api.runs(f"{entity}/{project}",
      {"group": group_name},
  )

  if state is None:
    runs = all_runs
  else:
    runs = [run for run in all_runs if run.state == state]
    
  if not runs:
      print("No runs found.")
      return pd.DataFrame(), []

  runs_json = [{"xid": run.id, **run.config} for run in runs]

  summary_df = pd.json_normalize(runs_json)
  sweep_vars = find_sweep_variables(summary_df)
  print(f"Sweep variables: {sweep_vars}")
  results = []
  for run in runs:
    
    result_single = run.history(samples=samples, keys=metric_names) 
    result_single['xid'] = run.id
    results.append(result_single)
    
  print("num runs", len(runs))
  if not results:
      return pd.DataFrame(), sweep_vars
      
  results_df = pd.concat(results)
  # Ensure merge is possible
  df = summary_df[sweep_vars + ['xid']].merge(results_df, on='xid')
  return df, sweep_vars

def _mean(data: pd.DataFrame, span: float, edge_tolerance: float = 0.):
  """Compute rolling mean of data via histogram, smooth endpoints.

  Args:
    data: pandas dataframe including columns ['x', 'y'] sorted by 'x'
    span: float in (0, 1) proportion of data to include.
    edge_tolerance: float of how much forgiveness to give to points that are
      close to the histogram boundary (in proportion of bin width).

  Returns:
    output_data: pandas dataframe with 'x', 'y' and 'stderr'
  """
  num_bins = np.ceil(1. / span).astype(np.int32)
  # Handle empty data
  if data.empty:
      return pd.DataFrame(dict(x=[], y=[], stderr=[]))
      
  count, edges = np.histogram(data.x, bins=num_bins)

  # Include points that may be slightly on wrong side of histogram bin.
  tol = edge_tolerance * (edges[1] - edges[0])

  x_list = []
  y_list = []
  stderr_list = []
  for i, num_obs in enumerate(count):
    if num_obs > 0:
      sub_df = data.loc[(data.x > edges[i] - tol)
                        & (data.x < edges[i + 1] + tol)]
      x_list.append(sub_df.x.mean())
      y_list.append(sub_df.y.mean())
      stderr_list.append(sub_df.y.std() / np.sqrt(len(sub_df))) # standardize for each bin (bins have different num of ssamples).

  return pd.DataFrame(dict(x=x_list, y=y_list, stderr=stderr_list))

def normal_lineplot(x, y, std = None, step_val_counts=None,
                      num_std=3,
                      span=0.005,
                      edge_tolerance=0.,
                      ax=None,
                      **kwargs):
    """Assuming all runs have the same logged number of steps, we average their metrics and plot 2 std deviation."""
    if std is None:
      data = pd.DataFrame(dict(x=x, y=y))
    else:
      data = pd.DataFrame(dict(x=x, y=y, std=std))

    ax = ax if ax is not None else plt.gca()

    if data.empty:
        return

    output_data = _mean(data, span, edge_tolerance)
    if output_data.empty:
        return
        
    output_data['ymin'] = output_data.y - num_std * output_data.stderr
    output_data['ymax'] = output_data.y + num_std * output_data.stderr
    
    # We remove label from kwargs for the fill_between to avoid duplicate legend entries if it was passed
    fill_kwargs = kwargs.copy()
    if 'label' in fill_kwargs:
        fill_kwargs.pop('label')
        
    ax.plot(output_data.x, output_data.y, **kwargs)

    ax.fill_between(
        output_data.x,
        output_data.ymin,
        output_data.ymax,
        alpha=0.1, **fill_kwargs)

    # Auto-scaling logic from notebook, slightly modified to be safer
    if not output_data.empty:
         ymin = min(output_data.ymin)
         ymax = max(output_data.ymax)
         if np.isfinite(ymin) and np.isfinite(ymax):
             ax.set_ylim([ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax)])

