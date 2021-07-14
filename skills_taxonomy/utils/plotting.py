import os
from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

# Load config
graph_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/graphs.yaml")
)

graph_output_path = str(PROJECT_DIR) + graph_config["GRAPH_OUTPUT_PATH"]


def save_fig(plot, fig_name, tight_layout=True, fig_extension="png", resolution=500):
    """Save current matplotlib plot as image with given file name.

    Parameters:

        fig_name (string):
        figure name

    Keyword arguments:

        tight_layout (Boolean):
        Set a tight layout for the figure (default = True)

        fig_extension (string):
        File format, e.g. "jpeg" will make the file end in .jpeg, default = "png"

        resolution (int):
        Resolution of saved figure (default = 500)

    Return: None"""

    # Set the file path
    path = os.path.join(graph_output_path, fig_name + "." + fig_extension)

    # Get tight layout if necessary
    if tight_layout:
        plot.tight_layout()

    # Save the figure in the right format and resolution
    plot.savefig(path, format=fig_extension, dpi=resolution)


def get_ISCO_node_colors(G, occ_isco_dict):
    """Color-encode nodes with top level of ISCO.

    Args:
        G (networkx.Graph):
        graph to color

    Return:
        node_colors (list):
        list with color encoding for nodes"""

    # Set node colors
    node_colors = []

    for node in G.nodes:

        # Get initial digit
        group = node[0]

        # If no digit
        if group not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:

            # Handle root node
            if node == "root":
                group = "10"

            # Handle occupation
            else:
                group = occ_isco_dict[node].str[0]

        # Append to node color list
        node_colors.append(int(group))

    return node_colors
