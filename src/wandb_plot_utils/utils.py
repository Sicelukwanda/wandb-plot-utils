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
