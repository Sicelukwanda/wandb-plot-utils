# wandb-plot-utils

Utilities for plotting Weights & Biases experiments.

## Installation

```bash
uv pip install -e .
```

## Usage

```python
import wandb_plot_utils as wpu

wpu.setup_style()

# Example usage
# df, sweep_vars = wpu.load_experiment_group_state(
#    group_name="experiment_group_name",
#    metric_names=["metric1", "metric2"],
#    entity="your_entity",
#    project="your_project"
# )
```
