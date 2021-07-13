# File: getters/isco_esco_data.py
"""Data getters for ISCO occupations and ESCO skills.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------


# Imports
import pandas as pd
import json

from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

# ---------------------------------------------------------------------------------

# Load config file for ISCO/ESCO data
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/isco_esco_data.yaml")
)


def get_occupations():
    """Load ISCO occupations.

    Return:
        Occupation information as DataFrame.
    """

    # Get path and load csv file
    occ_path = str(PROJECT_DIR) + config["OCCUPATIONS_PATH"]
    occupations_df = pd.read_csv(occ_path)

    return occupations_df


def get_skills():
    """Load ESCO skills.

    Return:
        Skills information as DataFrame.
    """

    # Get path and load csv file
    skills_path = str(PROJECT_DIR) + config["SKILLS_PATH"]
    skills_df = pd.read_csv(skills_path)

    return skills_df


def get_occup_skill_dict():
    """Load dict that maps occupations to skills.

    Return:
        Dict connecting occupations and skills.
    """

    # Get path
    occ_skill_dict_path = str(PROJECT_DIR) + config["OCC_SKILL_DICT_PATH"]

    # Load json file
    with open(occ_skill_dict_path, "r") as infile:
        occup_skills_dict = json.load(infile)

    return occup_skills_dict
