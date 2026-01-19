import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_experiment_group_state, normal_lineplot, setup_style
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    setup_style()
    
    # Ensure plot x and y are in metrics to fetch
    metrics = set(cfg.metrics)
    metrics.add(cfg.plot.x)
    
    # Handle best_so_far dependency
    if cfg.plot.y == "actor/best_so_far" and "actor/episode_return" not in metrics and "actor/best_so_far" not in metrics:
        metrics.add("actor/episode_return")
        
    print(f"Loading data for group: {cfg.group}...")
    df, sweep_vars = load_experiment_group_state(
        group_name=cfg.group,
        metric_names=list(metrics),
        entity=cfg.entity,
        project=cfg.project
    )
    
    if df.empty:
        print("No data found.")
        return

    # Post-processing special case for best_so_far
    if cfg.plot.y == "actor/best_so_far" and "actor/best_so_far" not in df.columns:
        if "actor/episode_return" in df.columns:
             df['actor/best_so_far'] = df.groupby('xid')['actor/episode_return'].cummax()
    
    print(f"Plotting {cfg.plot.y} vs {cfg.plot.x}...")
    
    plt.figure(figsize=(10, 6))
    
    if sweep_vars:
        print(f"Found sweep variables: {sweep_vars}. Plotting averaged over all variants.")
    
    normal_lineplot(
        x=df[cfg.plot.x], 
        y=df[cfg.plot.y],
        ax=plt.gca()
    )
    
    if cfg.plot.title:
        plt.title(cfg.plot.title)
    else:
        plt.title(f"{cfg.plot.y} vs {cfg.plot.x}")
        
    plt.xlabel(cfg.plot.x)
    plt.ylabel(cfg.plot.y)
    
    plt.gca().spines[['top', 'right']].set_visible(False)
    
    output_path = cfg.plot.output
    # Resolve user path if needed, though pyplot usually handles it? 
    # Hydra changes cwd, so relative paths might be tricky if user expects them relative to invocation dir.
    # Hydra creates a unique output dir per run. 
    # But for a simple tool, user might expect output in current dir.
    # By default hydra changes working directory. 
    # If user provides absolute path, it's fine. 
    # If relative, it goes into hydra run dir.
    # We should probably respect hydra's way or print where it is saved.
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
