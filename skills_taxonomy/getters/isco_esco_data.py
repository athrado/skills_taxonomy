# File: getters/isco_esco_data.py
"""Data getters for ISCO occupations and ESCO skills."""

import pandas as pd
import json

from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

# Load config
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/isco_esco_data.yaml")
)


def get_occupations() -> pd.DataFrame:
    """Load ISCO occupations.

    Returns:
        Occupation information as DataFrame.
    """

    occ_path = str(PROJECT_DIR) + config["OCCUPATIONS_PATH"]
    occupations_df = pd.read_csv(occ_path)

    return occupations_df


def get_skills() -> pd.DataFrame:
    """Load ESCO skills.

    Returns:
        Skills information as DataFrame.
    """

    skills_path = str(PROJECT_DIR) + config["SKILLS_PATH"]
    skills_df = pd.read_csv(skills_path)

    return skills_df


def get_occup_skill_dict() -> pd.DataFrame:
    """Load dict that maps occupations to skills.

    Returns:
        Dict connecting occupations and skills.
    """

    occ_skill_dict_path = str(PROJECT_DIR) + config["OCC_SKILL_DICT_PATH"]

    with open(occ_skill_dict_path, "r") as infile:
        occup_skills_dict = json.load(infile)

    return occup_skills_dict
