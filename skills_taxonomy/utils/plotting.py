# File: plotting.py
"""Plotting utils (so far only saving figures).

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
import os
from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

# ---------------------------------------------------------------------------------

# Load config
graph_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/graphs.yaml")
)

# Get graph output path
graph_output_path = str(PROJECT_DIR) + graph_config["GRAPH_OUTPUT_PATH"]


def save_fig(plot, fig_name, tight_layout=False, fig_extension="png", resolution=500):
    """Save current matplotlib plot as image with given file name.

    Parameter
    ----------
    fig_name: str
        figure name.

    tight_layout: bool
        Set a tight layout for the figure (default = True).

    fig_extension: str
        File format, e.g. "jpeg" will make the file end in .jpeg, default = "png".

    resolution: int
        Resolution of saved figure (default = 500).

    Return
    ----------
    None"""

    # Set the file path
    path = os.path.join(graph_output_path, fig_name + "." + fig_extension)

    # Get tight layout if necessary
    if tight_layout:
        plot.tight_layout()

    # Save the figure in the right format and resolution
    plot.savefig(path, format=fig_extension, dpi=resolution)
