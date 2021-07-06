from skills_taxonomy.getters import isco_esco_data

ESCO_occup_skills = isco_esco_data.get_occup_skill_dict()


def get_essential_skills(occupation):
    """Return essential skills for given occupation.

    Args:
        occupation (str) -- profession/occupation

    Return:
        skills (list) -- essential skills for profession"""

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
    """Return optional skills for given occupation.

    Args:
        occupation (str) -- profession/occupation

    Return:
        skills (list) -- optional skills for profession"""

    # Get optional skills for occupation
    try:
        optional_skills = ESCO_occup_skills[occupation]["_links"]["hasOptionalSkill"]
    except KeyError:
        return []

    # Get "title" for each
    optional_skills = [skill["title"] for skill in optional_skills]

    return optional_skills


def get_all_skills(occupation):
    """Return both optional and essential skills for given occupation.

    Args:
        occupation (str) -- profession/occupation

    Return:
        skills (list) -- all skills for profession"""

    essential_skills = get_essential_skills(occupation)
    optional_skills = get_optional_skills(occupation)

    return essential_skills + optional_skills


def get_skills(occupation, skill_type):
    """Return both optional and essential skills for given occupation.

    Args:
        occupation (str) -- profession/occupation

    Return:
        skills (list) -- all skills for profession"""

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
