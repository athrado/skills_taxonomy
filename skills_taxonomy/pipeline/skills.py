# File: getters/skills.py
"""Get essential and optional skills for given occupation.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
from skills_taxonomy.getters import isco_esco_data

# ---------------------------------------------------------------------------------

# Load dict that maps ESCO occupations to skills
ESCO_occup_skills = isco_esco_data.get_occup_skill_dict()


def get_essential_skills(occupation):
    """Get essential skills for given occupation.

    Parameters
    ----------
    occupation: string
        Occupation for which to get essential skills.

    Return
    ---------
    skills: list
        Essential skills for given occupation."""

    # Get essential skills for occupation
    try:
        essent_skills = ESCO_occup_skills[occupation]["_links"]["hasEssentialSkill"]
    except KeyError:
        return []

    # Get "title" for each
    essent_skills = [skill["title"] for skill in essent_skills]

    if essent_skills is None:
        return []

    return essent_skills


def get_optional_skills(occupation):
    """Get optional skills for given occupation.

    Parameters
    ----------
    occupation: string
        Occupation for which to get optional skills.

    Return
    ---------
    skills: list
        Optional skills for given occupation."""

    # Get optional skills for occupation
    try:
        optional_skills = ESCO_occup_skills[occupation]["_links"]["hasOptionalSkill"]
    except KeyError:
        return []

    # Get "title" for each
    optional_skills = [skill["title"] for skill in optional_skills]

    return optional_skills


def get_all_skills(occupation):
    """Get all skills (essential or optional) for given occupation.

    Parameters
    ----------
    occupation: string
        Occupation for which to get all skills.

    Return
    ---------
    skills: list
        All skills for given occupation."""

    essential_skills = get_essential_skills(occupation)
    optional_skills = get_optional_skills(occupation)

    return essential_skills + optional_skills


def get_skills(occupation, skill_type):
    """Get both optional and essential skills for given occupation.

    Parameters
    ----------
    occupation: string
        Occupation for which to get all skills.

    skill_type: 'all', 'essential', 'optional'
        Skill type to retrieve.

    Return
    ---------
    skills: list
        Skills for given occupation."""

    if skill_type == "all":
        return get_all_skills(occupation)
    elif skill_type == "essential":
        return get_essential_skills(occupation)
    elif skill_type == "optional":
        return get_optional_skills(occupation)
    else:
        raise IOError(
            "{} is not a known skill type, please select 'essential', 'optional' or 'all'.".format(
                skill_type
            )
        )


def main():
    """Testing skill retrieval functions."""

    # Testing
    print("Essential skills for technical director:")
    print(get_essential_skills("technical director"))
    print()
    print("Essential skills for air traffic safety technician:")
    print(get_essential_skills("air traffic safety technician"))
    print()
    print("Optional skills for air traffic safety technician:")
    print(get_optional_skills("air traffic safety technician"))
    print()
    print("All skills for air traffic safety technician:")
    print(get_all_skills("air traffic safety technician"))
    print()


if __name__ == "__main__":
    main()
