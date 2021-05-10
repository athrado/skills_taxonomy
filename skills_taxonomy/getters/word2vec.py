# File: getters/isco_esco_data
"""Load word2vec."""

from cookiecutter_training_julia import get_yaml_config, Path
import gensim
from gensim.models import KeyedVectors

# Load word2vec configs
word2vec_config = get_yaml_config(
    Path("cookiecutter_training_julia/config/pipeline/word2vec.yaml")
)

word2vec_path = word2vec_config["WORD2VEC_PATH"]


def load_word2vec() -> dict:
    """Load word2vec embeddings.

    Returns:
        Word2vec embeddings."""

    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    return word2vec
