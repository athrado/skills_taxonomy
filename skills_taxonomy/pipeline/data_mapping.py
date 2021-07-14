# File: getters/data_mapping.py
"""Connect skills, occupations and ISCO levels.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------


# Imports
from collections import defaultdict
from skills_taxonomy.pipeline import skills


# ---------------------------------------------------------------------------------

# Check skills for different occupation groups
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

# Invert dict
top_level_ID_dict = {v: k for k, v in ID_top_level_dict.items()}


class ISCO(object):
    """ISCO (International Standard Classification of Occupations) with different levels."""

    def __init__(self, isco):
        """Set up ISCO object.

        Parameters
        ----------
            isco: str, int
            ISCO level number
        """

        # ISCO as int or str
        self.int = int(isco)
        self.str = str(isco)

        # If only three digits, pad with 0 at the beginning
        if len(self.str) == 3:
            self.str = "0" + self.str

        # Levels in ISCO hierarchy
        self.level_1 = self.str[0]
        self.level_2 = self.str[:2]
        self.level_3 = self.str[:3]
        self.level_4 = self.str[:4]

    def similarity(self, other):
        """Compute similarity between two ISCO IDs.

        Parameter
        ----------
            other: ISCO instance
            ISCO ID to comapre with.

        Return
        ----------
            Similarity level: int"""

        # If identical
        if self.str == other.str:
            return 4

        # If first three levels identical
        elif self.str[:3] == other.str[:3]:
            return 3

        # If first two levels identical
        elif self.str[:2] == other.str[:2]:
            return 2

        # If only top level identical
        elif self.str[:1] == other.str[:1]:
            return 1

        # If no overlap in ISCO
        else:
            return 0

    def __str__(self):
        """Return string."""

        return str(self.str)


def get_data_links(occupation_df, skill_df):
    """Get connections between occupations, skills and ISCO levels
    as dictionaries.

    Parameters
    ----------
        occupation_df: pandas Dataframe
        Dataframe for occupations

        skill_df: pandas Dataframe
        Dataframe for skills

    Return
    ----------
        link_dict: dict
        Nested dictionary for linking occupations, skills and ISCO levels.

        missing_dict: dict
        Nested dictionary holding lists with missing skill and ISCO entities
    """

    # Initialize skill dicts
    skill_isco_dict = defaultdict(list)
    skill_occu_dict = defaultdict(list)

    # Initialize ISCO dicts
    isco_skill_dict = defaultdict(list)
    isco_occ_dict = defaultdict(list)

    # Initialize occupation dicts
    occ_skill_dict = defaultdict(list)
    occ_isco_dict = {}

    # Occupations without essential/optional skills
    missing_essent_skills_occ = []
    missing_opt_skills_occ = []
    missing_isco_skills = []

    # Get occupations and skills
    occupation_set = sorted(set(occupation_df["preferredLabel"].values))
    skill_set = sorted(set(skill_df["preferredLabel"].values))

    # For each occupation
    for occupation in occupation_set:

        # Get occupation data
        occupation_data = occupation_df.loc[
            occupation_df["preferredLabel"] == occupation
        ]

        # Get ISCO ID as ISCO instance
        isco = occupation_data["iscoGroup"].item()
        isco = ISCO(isco)

        # Get skills (or mark if missing)
        essential_skills = skills.get_essential_skills(occupation)
        optional_skills = skills.get_optional_skills(occupation)

        if not essential_skills:
            missing_essent_skills_occ.append(occupation)
            continue

        if not optional_skills:
            missing_opt_skills_occ.append(occupation)

        # For each skill, save ISCO and occupation
        for skill in essential_skills:
            skill_isco_dict[skill].append(isco)
            skill_occu_dict[skill].append(occupation)

        # For ISCO ID, save skills and occupation
        isco_skill_dict[isco.str] += essential_skills
        isco_occ_dict[isco.str].append(occupation)

        # For occupation, save skills and ISCO IDs
        occ_skill_dict[occupation] += essential_skills
        occ_isco_dict[occupation] = isco

    # Track missing skills
    for skill in skill_set:
        if skill not in skill_isco_dict.keys():
            missing_isco_skills.append(skill)

    # Set up dict for linking dicts
    link_dict = {
        "skill_to_ISCO": skill_isco_dict,
        "skill_to_occup": skill_occu_dict,
        "ISCO_to_skill": isco_skill_dict,
        "ISCO_to_occup": isco_occ_dict,
        "occup_to_skill": occ_skill_dict,
        "occup_to_ISCO": occ_isco_dict,
    }

    # Set up dict for missing lists
    missing_dict = {
        "occ_with_miss_essent_skills": missing_essent_skills_occ,
        "occ_with_miss_opt_skills": missing_opt_skills_occ,
        "skills_with_miss_ISCO": missing_isco_skills,
    }

    return link_dict, missing_dict


def get_skills_for_sector(sector, isco_set, isco_skill_dict):
    """Get all skills for a given sector.

    Parameters
    ----------
        isco_set: list
        List of ISCO IDs.

        isco_skill_dict: dict
        Dict that maps ISCO ID to skills required for occupations with that ID.

        sector: str, int
        Sector as string or sector ID as int.

    Return
    ----------
        sector_skills: list
        List of skills required in given sector.
    """

    # Get top level sector name
    if isinstance(sector, str):
        top_level = top_level_ID_dict[sector]
    else:
        top_level = sector

    # Get sector skills
    sector_skills = [
        isco_skill_dict[str(isco)]
        for isco in isco_set
        if int(ISCO(isco).level_1) == top_level
    ][0]

    # Get sorted and unique list
    sector_skills = sorted(set(sector_skills))

    return sector_skills
