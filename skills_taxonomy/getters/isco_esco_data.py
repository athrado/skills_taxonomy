# File: getters/isco_esco_data
"""Data getters for ISCO occupations and ESCO skills."""

import pandas as pd
import json


def get_occupations():
    """Load ISCO occupations.

    Returns:
        Occupation information as DataFrame.
    """

    occupations_df = pd.read_csv("./inputs/data/occupations_en.csv")

    return occupations_df


def get_skills():
    """Load ESCO skills.

    Returns:
        Skills information as DataFrame.
    """

    skills_df = pd.read_csv("./inputs/data/skills_en.csv")

    return skills_df


def get_occup_skill_dict():
    """Load dict that maps occupations to skills.

    Returns:
        Dict connecting occupations and skills.
    """

    with open("./inputs/data/ESCO_occup_skills.json", "r") as infile:
        occup_skills_dict = json.load(infile)

    return occup_skills_dict
