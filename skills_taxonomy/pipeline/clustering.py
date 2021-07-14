# File: getters/clustering.py
"""Clustering of ESCO skills.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD

from skills_taxonomy.pipeline import data_mapping
import skills_taxonomy.utils.plotting as plotting

# ---------------------------------------------------------------------------------


def plot_explained_variance(dim_reduction, title):
    """Plot percentage of variance explained by each of the selected components
    after performing dimensionality reduction (e.g. PCA, LSA).

    Parameters
    ----------
        dim_reduction: sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD
        Dimensionality reduction on features with PCA or LSA.

        title: str
        Title for saving plot.

    Return: None"""

    # Explained variance ratio (how much is covered by how many components)

    # Per component
    plt.plot(dim_reduction.explained_variance_ratio_)
    # Cumulative
    plt.plot(np.cumsum(dim_reduction.explained_variance_ratio_))

    # Assign labels and title
    plt.xlabel("Dimensions")
    plt.ylabel("Explained variance")
    plt.title("Explained Variance Ratio by Dimensions " + title)

    # Save plot
    plotting.save_fig(plt, "Explained Variance Ratio by Dimensions " + title)

    # Show plot
    plt.show()


def dimensionality_reduction(
    features,
    dim_red_technique="LSA",
    lsa_n_comps=90,
    pca_expl_var_ratio=0.90,
    random_state=42,
):
    """Perform dimensionality reduction on given features.

    Parameters
    ----------
        features: np.array
        Original features on which to perform dimensionality reduction.

        dim_red_tequnique: 'LSA', 'PCA', default='LSA'
        Dimensionality reduction technique.

        lsa_n_comps: int, default=90
        Number of LSA components to use.

        pca_expl_var_ratio: float (between 0.0 and 1.0), default=0.90
        Automatically compute number of components that fulfill given explained variance ratio (e.g. 90%).

        random_state: int, default=42
        Seed for reproducible results.

    Return
    ---------
        lsa_transformed or pca_reduced_features: np.array
        Dimensionality reduced features."""

    if dim_red_technique.lower() == "lsa":

        # Latent semantic analysis (truncated SVD)
        lsa = TruncatedSVD(n_components=lsa_n_comps, random_state=random_state)
        lsa_transformed = lsa.fit_transform(features)

        plot_explained_variance(lsa, "LSA")

        print("Number of features after LSA: {}".format(lsa_transformed.shape[1]))

        return lsa_transformed

    elif dim_red_technique.lower() == "pca":

        # Principal component analysis
        pca = PCA(random_state=random_state)

        # Transform features
        pca_transformed = pca.fit_transform(features)

        plot_explained_variance(pca, "PCA")

        # Get top components (with combined explained variance ratio of e.g. 90%)
        pca_top = PCA(n_components=pca_expl_var_ratio)
        pca_reduced_features = pca_top.fit_transform(features)

        # print
        print("Number of features after PCA: {}".format(pca_reduced_features.shape[1]))

        return pca_reduced_features

    else:
        raise IOError(
            "Dimensionality reduction technique '{}' not implemented.".format(
                dim_red_technique
            )
        )


def k_means_clustering(features, n_clusters=15, random_state=42):
    """Cluster given features into given number of clusters using k-means clustering.

    Parameters
    ----------
        features: np.array
        Features for clustering.

        n_clusters: int, default=15
        Number of clusters.

        random_state: int, default=42
        Seed for reproducible results.

    Return
    ---------
        kmeans: sklearn.cluster.KMeans
        Result of k-means clustering on given features."""

    # K-means Clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_jobs=None,
        precompute_distances="auto",
    )
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)

    # Plot cluster distribution
    plt.scatter(features[:, 0], features[:, 1], c=y_kmeans, s=50, cmap="viridis")

    # Plot red centers
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.5)
    plotting.save_fig(plt, "K-means clustering")

    return kmeans


def get_updated_skill_df(skills_df, skills_of_interest, df_dict, kmeans):
    """Update skill dataframe with with Cluster ID and document frequency.

    Parameters
    ----------
        skills_df: pandas.DataFrame
        Original skill dataframe.

        skills_of_interest: list
        List of skills of interested (as strings)

        df_dict: dict
        Document frequency dict for every ISCO node.

        kmeans: sklearn.cluster.KMeans
        Results from k-means clustering.

    Return
    ---------
        updated_skill_df: pandas.DataFrame
        Updated skill dataframe with new information.

        skill_mapping_dicts: dict
        Dictionarys for mapping skills and descriptions to respetivec clusters and vice versa."""

    # Get document frequency for every skill
    df_values = [df_dict[skill] for skill in skills_of_interest]

    # Get descriptions for skills
    descriptions = [
        skills_df.loc[skills_df["preferredLabel"] == skill]["description"].item()
        for skill in skills_of_interest
    ]

    # Set dictionaries for easier access
    desc_skill_dict = {
        descriptions[i]: skills_of_interest[i] for i in range(len(skills_of_interest))
    }
    skill_cluster_dict = {
        skills_of_interest[i]: kmeans.labels_[i] for i in range(len(skills_of_interest))
    }

    # Add the data to the dataframe
    updated_skill_df = pd.DataFrame()
    updated_skill_df["skill"] = skills_of_interest
    updated_skill_df["description"] = descriptions
    updated_skill_df["cluster"] = kmeans.labels_
    updated_skill_df["DF"] = df_values

    n_clusters = len(list(set(kmeans.labels_)))

    # Get cluster_skill_dict
    cluster_skill_dict = {
        cluster: [
            skill
            for skill in updated_skill_df[updated_skill_df.cluster == cluster]["skill"]
        ]
        for cluster in range(n_clusters)
    }

    # Sort the skills by document frequency (DF)
    updated_skill_df = updated_skill_df.sort_values("DF", ascending=False)

    # Set dict for mapping skills to cluster and vice versa
    skill_mapping_dicts = {
        "desc to skill": desc_skill_dict,
        "skill to cluster": skill_cluster_dict,
        "cluster to skill": cluster_skill_dict,
    }

    return updated_skill_df, skill_mapping_dicts


def visualise_tf_idf_for_skill(G_ISCO, tf_idf, ISCO_nodes, skills_of_interest, skill):
    """Visualise TF-IDF

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
    plotting.save_fig(plt, "TF-IDF on for {}".format(skill))

    plt.show()
