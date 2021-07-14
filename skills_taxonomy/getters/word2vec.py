# File: getters/isco_esco_data
"""Load word2vec data.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR
from gensim.models import KeyedVectors

# ---------------------------------------------------------------------------------

# Load word2vec configs
word2vec_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/word2vec.yaml")
)

# Get path for word2vec data
word2vec_path = word2vec_config["WORD2VEC_PATH"]


def load_word2vec():
    """Load word2vec embeddings.

    Return
    ----------
        word2vec (dict):
        Word2vec embeddings with words as key and embeddings as values."""

    # Load embeddings
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    return word2vec
