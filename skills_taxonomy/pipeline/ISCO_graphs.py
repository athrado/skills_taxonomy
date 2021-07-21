# File: getters/ISCO_graphs.py
"""Create ISCO graphs for visualisation and TF-IDF computation.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pickle

from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

from skills_taxonomy.pipeline import data_mapping
from skills_taxonomy.pipeline.data_mapping import ISCO
import skills_taxonomy.utils.plotting as plotting

# ---------------------------------------------------------------------------------

# Load graphs config
graph_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/graphs.yaml")
)

# Get TF-IDF output path
tfidf_output_path = str(PROJECT_DIR) + graph_config["TDIDF_OUTPUT_PATH"]

# Occupation sector labels
ID_top_level_dict = {
    0: "Armed forces occupations",
    1: "Managers",
    2: "Professionals",
    3: "Technicians and associate professionals",
    4: "Clerical support workers",
    5: "Service and sales workers",
    6: "Skilled agricultural, forestry and fishery workers",
    7: "Craft and related trades workers",
    8: "Plant and machine operators and assemblers",
    9: "Elementary occupations",
}

# ---------------------------------------------------------------------------------


def create_circular_ISCO_graph(
    occupation_set,
    link_dict,
    reduced_occupation_set=False,
    only_one_leaf_node=False,
):
    """Visualise ISCO hierarchy as circular tree graph.

    Parameters
    ----------
    occupation set: list
        List of occupations of interest.

    link_dict: dict
        Dict that holds further dicts for connecting occupations, skills and ISCO skills.

    reduced_occupation_set: bool, default=False
        Only use the first 100 occupations of occupation_set.

    only_one_leaf_node: bool, default=False
        Only display one leaf node (at outermost level) for better visibility.


    Return
    ----------
    G_ISCO: networkx.Graph
        Directed graph representing the ISCO hierarchy.
    """

    # Get mapping dicts
    isco_occ_dict = link_dict["ISCO_to_occup"]
    occ_isco_dict = link_dict["occup_to_ISCO"]

    # If only displaying reduced set
    if reduced_occupation_set == True:
        occupations_to_process = occupation_set[:100]

    else:
        # Use all occupations
        occupations_to_process = occupation_set

    # Get ISCO set for given occupations
    ISCO_set = list(set([occ_isco_dict[occ] for occ in occupations_to_process]))

    # Set up a graph with nodes
    G_ISCO = nx.DiGraph()

    # For each collected ISCO ID
    for isco in ISCO_set:

        # Get ISCO as string
        isco = isco.str

        # Get occupations with this ISCO
        occupations = isco_occ_dict[isco]

        # Add hierarchy edges
        G_ISCO.add_edge("root", isco[0])
        G_ISCO.add_edge(isco[0], isco[:2])
        G_ISCO.add_edge(isco[:2], isco[:3])
        G_ISCO.add_edge(isco[:3], isco)

        if only_one_leaf_node:
            # Get only one exmplary occupation as outermost level
            G_ISCO.add_edge(isco, occupations[0])

        else:
            # Add all occupation as leaves
            for occupation in occupations:
                G_ISCO.add_edge(isco, occupation)

    return G_ISCO


def get_n_leaf_nodes(G, node):
    """Get number of leaf nodes for given node.

    Parameter
    ----------
    G: network.Graph
        Graph that is used to count leaf nodes.

    node: network.Graph.node
        Node of interest within graph G for which to count leaf nodes.

    Return
    ----------
    n_leaf_nodes: int
        Number of leaf nodes for given node in G."""

    # Get descendants for given node in G
    descendants = nx.descendants(G, node)

    # Get leaf nodes (outdegree = 0)
    leaf_nodes = [desc for desc in descendants if G.out_degree(desc) == 0]

    # Number of leaves
    n_leaf_nodes = len(leaf_nodes)

    return n_leaf_nodes


def get_tf(G, node, skill, leaf_skill_dict):
    """Compute 'term frequency' for given skill at specific node in graph.
    Term frequency (tf) in this context means skill frequency by number of (ISCO) nodes.

    Parameters
    ----------
    G: network.Graph
        Graph for which to compute term frequency.

    node: network.Graph.node
        Node of interest within graph G.

    skill: str
        Skill of interest for which to compute term frequency.

    leaf_skill_dict: dict
        Dictionary that maps leaf to skill set.

    Return
    ----------
    tf: float
        Term frequency for given skill and node."""

    # Get descendants for given node in G
    descendants = nx.descendants(G, node)

    # Get leaf nodes (outdegree = 0)
    leaf_nodes = [desc for desc in descendants if G.out_degree(desc) == 0]

    # Get 'term leaves' (i.e. occupation leaves which require specific skill)
    term_leaves = [leaf for leaf in leaf_nodes if skill in leaf_skill_dict[leaf]]

    # Compute frequency (over all possible leaves)
    tf = len(term_leaves) / len(leaf_nodes)

    # Return term frequency
    return tf


def map_ISCO_nodes_to_feat_indices(isco_set):
    """
    Map ISCO IDs to feature indices.

    Parameters
    ----------
    isco_set: list
        List of ISCO IDs as strings.

    Return
    ----------
    ISCO_idx_dict: dict
        Dictonary that maps ISCO node to feature index.

    ISCO_nodes_dict: dict
        Dictonary that gives feature indices for nodes up to given ISCO level.

    ISCO_level_n_dict: dict
        Dictionary that gives number of nodes up to given ISCO level."""

    # Get ISCO set
    ISCOs = [ISCO(isco) for isco in isco_set]

    # Get all ISCO levels (1 to 4 digits)
    first_level = sorted(set([isco.level_1 for isco in ISCOs]))
    second_level = sorted(set([isco.level_2 for isco in ISCOs]))
    third_level = sorted(set([isco.level_3 for isco in ISCOs]))
    fourth_level = sorted(set([isco.level_4 for isco in ISCOs]))

    # Get numer of ISCO level nodes
    n_ISCO_nodes = (
        len(first_level) + len(second_level) + len(third_level) + len(fourth_level)
    )

    # Set up an ISCO level to index dict
    ISCO_idx_dict = {}

    # Set index for all first level ISCOs
    for i, first in enumerate(first_level):
        ISCO_idx_dict[first] = i

    # Set index for all second level ISCOs
    for i, second in enumerate(second_level):
        j = i + len(first_level)
        ISCO_idx_dict[second] = j

    # Set index for all third level ISCOs
    for i, third in enumerate(third_level):
        j = i + len(first_level) + len(second_level)
        ISCO_idx_dict[third] = j

    # Set index for all full ISCOs
    for i, fourth in enumerate(fourth_level):
        j = i + len(first_level) + len(second_level) + len(third_level)
        ISCO_idx_dict[fourth] = j

    # Number of features for different levels
    n_ISCO_up_to_third_level = len(first_level) + len(second_level) + len(third_level)
    n_ISCO_up_to_sec_level = len(first_level) + len(second_level)

    # Dict with number of nodes up to given ISCO level
    ISCO_level_n_dict = {
        4: n_ISCO_nodes,
        3: n_ISCO_up_to_third_level,
        2: n_ISCO_up_to_sec_level,
        1: len(first_level),
    }

    # Dict with feature indices for nodes up to given ISCO level.
    ISCO_nodes_dict = {
        1: first_level,
        2: first_level + second_level,
        3: first_level + second_level + third_level,
        4: first_level + second_level + third_level + fourth_level,
    }

    return ISCO_idx_dict, ISCO_nodes_dict, ISCO_level_n_dict


def compute_tf_idf(
    skills_of_interest,
    n_ISCO_nodes,
    G_ISCO,
    ISCO_idx_dict,
    link_dict,
    pre_computed=True,
):
    """Compute the TF-IDF score for every node in the ISCO graph.
    More info on TF-IDF below parameter documentation.

    Parameters
    ----------
    skills_of_interest: list
        List of skills of interest as strings.

    n_ISCO_nodes: int
        Number of ISCO nodes (for setting up feature array).

    G_ISCO: networkx.Graph
        Circular ISCO graph for TF-IDF computation.

    ISCO_idx_dict: dict
        Mapping of ISCO to feature index to enable saving node features in ordered way.

    link_dict: dict
        Dictionary holding other dictionaries linking occupations, skills and ISCO IDs.

    pre_computed: bool, default=True
        If set to True, use load and return saved tf_idf and df_dict.

    Return
    ----------
    tf_idf: np.array
        Feature array holding TF-IDF scores for every node in ISCO graph.

    df_dict: dict
        Dictionary holding DF (document frequency) value for every ISCO node.

    Background Info on TF-IDF:
    We want to measure how specific or general a given skill is and
    how much it required for each ISCO node.
    This task is similar to the task of determining the relevance
    of a word in a document, with regards to its
    use in the entire set of documents. This common NLP task is usually
    resolved with the statistical method tf-idf.
    tf stands for term frequency, idf stands for inversed document frequency.
    The tf reprents the frequency of a the word in the given document,
    while the df represents in how many documents this word appears.
    Thus, the inversed df gives higher relevance to words that are not very general
    and only occur in few documents.
    In our case, the term frequency represents the frequency
    of a specific skill requirement at any ISCO node. Using the example above,
    skill A is relevant for two occupations leading up to ISCO node 512.
    The frequency is computed by counting the occurences (i.e. 2)
    and diving it by the highest possible count, i.e. the number of leaf nodes
    for ISCO node 512 (all downstream occupation nodes at the outermost level).
    The term frequency measures how relevant a skill is at given ISCO node
    in comparison to other skills.
    The document frequency represents here the number of ISCO nodes that are affected
    by the skill requirement. The additional association of skill A with ISCO ID 5123 affects
    the following nodes at different levels: 5123, 512, 51, 5. The association with 5125
    adds only one new node to this set: 5125 itself. Thus, if skill A is spread more widely
    over the ISCO graph and is required for occupations with ISCO IDs of various top level categories,
    the document frequency will be higher than if the skill only affects occupations
    in the same part of the ISCO graph. Thus, the document frequency computes
    how widely applicable or required a specific skill is across the ISCO graph,
    with higher values indicating more general skills and lower values more specific skills.
    For the computation of the tf-idf, we use the inversed document frequency in order
    to give more weight to specific skills. If a skill is very general,
    it should not affect the ISCO node as much. Thus, the tf-idf in regards
    to the skill distribution over the ISCO graph measures how relevant a skill is
    at a given ISCO node."""

    if pre_computed:

        # Load pre_computed features
        tf_idf = np.load(tfidf_output_path + "tf_idf.npy")

        # Get document frequency values for skills
        with open(tfidf_output_path + "df_dict.pickle", "rb") as input_file:
            df_dict = pickle.load(input_file)

        return tf_idf, df_dict

    # Get linking dictionaries
    occ_skill_dict = link_dict["occup_to_skill"]
    skill_isco_dict = link_dict["skill_to_ISCO"]

    # Set up a skill array (n skills x ISCO nodes)
    skill_array = np.zeros((len(skills_of_interest), n_ISCO_nodes))

    # Set up tf and df array with same shape
    tf_array = np.zeros((skill_array.shape))
    df_array = np.zeros((skill_array.shape))
    df_dict = {}

    # For each skill
    for i, skill in enumerate(skills_of_interest):

        # Get occupations and ISCO list
        isco_list = skill_isco_dict[skill]

        # For each ISCO ID
        for isco in isco_list:

            # Get nodes and indices
            nodes_indices = [
                (isco.level_1, ISCO_idx_dict[isco.level_1]),
                (isco.level_2, ISCO_idx_dict[isco.level_2]),
                (isco.level_3, ISCO_idx_dict[isco.level_3]),
                (isco.level_4, ISCO_idx_dict[isco.level_4]),
            ]

            # Only process levels with digit (do not process missing input levels)
            nodes_indices = nodes_indices[: len(isco.str)]

            # For each node, save tf at given index in array
            for node, idx in nodes_indices:

                # Get term frequency
                tf = get_tf(G_ISCO, node, skill, occ_skill_dict)
                tf_array[i, idx] = tf

                # Mark whether node is affect by skill
                skill_array[i, idx] = 1.0

        # Get document frequency
        df = np.sum(skill_array[i, :])
        df_array[i, :] = df
        df_dict[skill] = df

    # Get number of nodes
    N = skill_array.shape[1]

    # Compute tf-idf
    tf_idf = tf_array * np.log(N / df_array)

    # Save tf-idf
    np.save(tfidf_output_path + "tf_idf.npy", tf_idf)

    # Save document frequency
    with open(tfidf_output_path + "df_dict.pickle", "wb") as output_file:
        pickle.dump(df_dict, output_file)

    return tf_idf, df_dict


def get_ISCO_node_colors(G, occ_isco_dict):
    """Color-encode nodes with top level of ISCO.

    Parameters:
    -------------

    G: networkx.Graph
        Graph to color.

    Return
    -------------

    node_colors: list
        List with color encoding for nodes."""

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


def visualise_tf_idf_for_skill(G_ISCO, tf_idf, ISCO_nodes, skills_of_interest, skill):
    """Visualise TF-IDF on ISCO graph for given skill.

    Parameters
    ----------
    G_ISCO: networkx.Graph
        Graph representing ISCO hierarchy.

    tf_idf: np.array
        TF-IDF scores as features.

    ISCO_nodes: list
        List of feature indices referring to ISCO nodes.

    skills_of_interest: list
        List of skills (as strings).

    skill: str
        Skill for which to visualise TF-IDF scores.

    Return
    ---------
    None"""

    # Get TF-IDF scores for skill of interest
    skill_idx = skills_of_interest.index(skill)
    skill_tfidf = tf_idf[skill_idx, :]

    # Set TF-IDF dict for ISCO nodes
    node_idf_dict = dict(zip(ISCO_nodes, skill_tfidf))
    node_idf_dict["root"] = 0.0

    # Fill up with 0.0 for missing nodes
    for node in G_ISCO.nodes:
        if node not in node_idf_dict.keys():
            node_idf_dict[node] = 0.0

    # Set node weights based on TF-IDF
    node_weights = [node_idf_dict[node] for node in G_ISCO.nodes]

    # Normalise
    min_node = min(node_weights)
    max_node = max(node_weights)
    node_weights = [((x - min_node) / (max_node - min_node)) for x in node_weights]

    # Only display top level sector labels
    label_dict = data_mapping.ID_top_level_dict
    node_labels = {
        label: label_dict[int(label)] if len(label) <= 1 else ""
        for label in G_ISCO.nodes
    }

    # Make top level sector nodes slightly larger
    node_size = [750 if len(label) <= 1 else 200 for label in G_ISCO.nodes]

    # Set figure size and circular pos
    plt.figure(figsize=(15, 15))
    circular_pos = graphviz_layout(G_ISCO, prog="twopi", args="")

    # Draw graph
    nx.draw(
        G_ISCO,
        circular_pos,
        node_size=node_size,
        alpha=1,
        edge_color="gray",
        node_color=node_weights,
        font_size=8,
        cmap=cm.Reds,
        with_labels=True,
        labels=node_labels,
        edgecolors="black",
    )
    plt.axis("equal")

    # Save plot
    plotting.save_fig(plt, "TF-IDF for skill {}".format(skill))

    plt.show()
