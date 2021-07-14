# File: getters/build_taxonomy_branches.py
"""Build taxonomy branches (hierarchical triples and synonyms).

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import cm

from skills_taxonomy.pipeline import word_embeddings
from skills_taxonomy.utils import plotting

# ---------------------------------------------------------------------------------


def get_cosine_similarity(string_1, string_2, WORD2VEC):
    """Get cosine similarity for two strings.

    Parameters
    ----------
        string_1: str
        First word/phrase to embed.

        string_2: str
        Second word/phrase to embed.

        WORD2VEC: dict
        Dictionary with word2vec embeddings for words.

    Return
    ---------
        cosim: float
        Cosine similarity between two embedded strings."""

    # Embed decriptions using word2vec embeddings
    emb_desc_1 = word_embeddings.embed_phrase(WORD2VEC, string_1).reshape(1, -1)
    emb_desc_2 = word_embeddings.embed_phrase(WORD2VEC, string_2).reshape(1, -1)

    # Compute cosine similarity for both embeddings
    cosim = cosine_similarity(emb_desc_1, emb_desc_2)[0][0]

    return cosim


def get_branch_tuples(ranked_skill_triple, df_dict):
    """Get hierarchical skill triples.

    Parameters
    ----------
        ranked_skill_triple: tuple
        Ranked skill triples.

        df_dict: dict
        Document frequency dictionary

    Return
    ---------
        synonym_skills: tuple
        Tuple of skills that are synonyms

        skill_hierarchy: tuple
        Hierarchically ordered triple of skills."""

    synonym_skills = []
    skill_hierarchy = []

    # Get ranked document frequeuncy for triple
    ranked_skill_triple_df = [df_dict[elem] for elem in ranked_skill_triple]

    # Set three skills and dfs for comparison
    first_df, second_df, third_df = ranked_skill_triple_df
    first_skill, second_skill, third_skill = ranked_skill_triple

    # If df for all skills is the same, they are on the same level
    if (first_df == second_df) and (second_df == third_df):

        # Save the skills as similar and on same level
        synonym_skills.append((first_skill, second_skill))
        synonym_skills.append((second_skill, third_skill))
        synonym_skills.append((third_skill, first_skill))

    # If first two skills on same level, save connection between them to last
    elif first_df == second_df:

        if third_df < second_df:

            # Save skill hierachy
            skill_hierarchy.append((first_skill, third_skill))
            skill_hierarchy.append((second_skill, third_skill))

        else:

            # Save skill hierachy
            skill_hierarchy.append((third_skill, first_skill))
            skill_hierarchy.append((third_skill, second_skill))

        # First and second skill are similar and on same level
        synonym_skills.append((first_skill, second_skill))

    # If last two skills on same level, save connection between them to first
    elif second_df == third_df:

        if first_df < second_df:

            # Save skill hierachy
            skill_hierarchy.append((first_skill, second_skill))
            skill_hierarchy.append((first_skill, third_skill))

        else:
            # Save skill hierachy
            skill_hierarchy.append((second_skill, first_skill))
            skill_hierarchy.append((third_skill, first_skill))

        # First and second skill are similar and on same level
        synonym_skills.append((second_skill, third_skill))

    else:
        # Save skill hierarchy
        skill_hierarchy.append((first_skill, second_skill, third_skill))

    return synonym_skills, skill_hierarchy


def get_skill_tax_branches(
    clusters_of_interest,
    skills_df,
    df_dict,
    desc_skill_dict,
    WORD2VEC=None,
):
    """Get skill taxonomy branches, in form of hierarchically ordered triples and synonym tuples.

    Parameters
    ----------
        clusters_of_interest: list
        List of cluster labels as integers to inspect.

        skills_df: pandas.DataFrame
        Dataframe for skills with cluster and DF information.

        desc_skill_dict: dict
        Dictionary mapping descriptions to skills.

        WORD2VEC: dict, default=None
        Dictionary that gives word2vec embeddings for words.

    Return
    ---------
        cosim: float
        Cosine similarity between two embedded strings."""

    # Skill hierarchy tuples
    skill_hierarchy = []

    # Similar skills tuples
    synonym_skills = []

    if WORD2VEC is None:
        WORD2VEC = word_embeddings.load_word2vec()

    # For each cluster, extract taxonomy parts (we only use a selection)
    for cluster in clusters_of_interest:

        # Get all skills and descriptions for this cluster
        skills = [skill for skill in skills_df[skills_df.cluster == cluster]["skill"]]
        descriptions = [
            desc for desc in skills_df[skills_df.cluster == cluster]["description"]
        ]

        # Initialize a nested cosine similarity dict
        cosim_dict = defaultdict(dict)

        # For each description pair, compute cosine similarity
        for desc_1, desc_2 in itertools.combinations(descriptions, 2):

            cosim = get_cosine_similarity(desc_1, desc_2, WORD2VEC)

            # Get skill (via description)
            skill_1 = desc_skill_dict[desc_1]
            skill_2 = desc_skill_dict[desc_2]

            # Save cosine similarity in nested dict
            cosim_dict[skill_1][skill_2] = cosim

        # For each skill
        for skill in skills:

            # Retrieve cosine similarity dict for skill
            skill_sim_dict = cosim_dict[skill]

            # Get two most similar skills
            most_similar_skills = [
                k
                for k, v in sorted(
                    skill_sim_dict.items(), key=lambda item: item[1], reverse=True
                )
            ][:2]

            # Rank skill and most similar skills by document frequency
            ranked_skill_triple = sorted(
                most_similar_skills + [skill], key=lambda k: df_dict[k], reverse=True
            )
            # Discard if triple is not complete
            if len(ranked_skill_triple) != 3:
                continue

            new_synonym_skills, new_skill_hierarchy = get_branch_tuples(
                ranked_skill_triple, df_dict
            )

            # Add skill hierarchy tuples
            synonym_skills += new_synonym_skills
            skill_hierarchy += new_skill_hierarchy

    return skill_hierarchy, synonym_skills


def visualise_taxonomy_branches(
    skill_branches,
    skill_to_cluster_dict,
    directed_graph=True,
    every_nth_sample=5,
):

    """Visualise taxonomy branches.
    Ideally visualise hierarchical tuples with a directed graph

    Parameters
    ----------
        skill_branches: tuple
        Ranked skill triples.

        skill_to_cluster_dict: dict
        Document frequency dictionary

        directed_graph: bool, default=True
        Plot a directed graph, ideal for hierachical triples.

        every_nth_sample: int, default=5
        Only display every n-th sample.

    Return
    ---------
        synonym_skills: tuple
        Tuple of skills that are synonyms

        skill_hierarchy: tuple
        Hierarchically ordered triple of skills."""

    if directed_graph:
        # Set up a directed Graph
        G_skills = nx.DiGraph()
    else:
        G_skills = nx.Graph()

    # Add skill branches
    for skills in skill_branches[::every_nth_sample]:

        # For each connection, add an edge
        for i in range(len(skills) - 1):
            G_skills.add_edge(skills[i], skills[i + 1])

    # Set node colors
    node_colors = [skill_to_cluster_dict[node] for node in G_skills.nodes]

    # Set figure size
    plt.figure(figsize=(15, 15))

    # Draw plot
    pos = nx.spring_layout(G_skills, k=1)
    nx.draw(
        G_skills,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=cm.Pastel2,
        node_size=500,
    )

    # Plat taxonomy graph
    plotting.save_fig(plt, "Taxonomy_graph")
