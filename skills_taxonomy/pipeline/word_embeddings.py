# File: getters/word_embeddings.py
"""Word embeddings.

Created April 2021
@author: Julia Suter
Last updated on 13/07/2021
"""

# ---------------------------------------------------------------------------------

# Imports
import re
import numpy as np

from gensim.models import KeyedVectors
from textblob import TextBlob, Word

from skills_taxonomy import get_yaml_config, Path, PROJECT_DIR

# ---------------------------------------------------------------------------------

# Load word2vec configs
word2vec_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/skills_taxonomy/config/pipeline/word2vec.yaml")
)

# Get word2vec path and number of embedding dimensions
word2vec_path = str(PROJECT_DIR) + word2vec_config["WORD2VEC_PATH"]
EMBED_DIM = word2vec_config["EMBED_DIM"]


def load_word2vec() -> dict:
    """Load word2vec embeddings.

    Returns:
        Word2vec embeddings."""

    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    return word2vec


def embed_phrase(WORD2VEC, phrase, EMBED_DIM=300):
    """Word2vec embedding for phrase.

    WORD2VEC: dict
        Dictioanry holding (300-dim) word2vec embeddings.

    phrase: str
        Word or phrase to embed.

    EMBED_DIM: int
        Embedding dimensions to create embeddings array."""

    # Correct phrase, get tokens and length of phrase
    phrase = re.sub("_", " ", phrase)

    blob_phrase = TextBlob(phrase)
    tokens = [elem[0] for elem in blob_phrase.tags]
    length_phrase = len(tokens)

    # If no words, fill with zeros
    if length_phrase == 0:
        res = np.zeros((1, EMBED_DIM))  # glove dim
        emb = np.mean(res, axis=0)
        return emb

    # Prepare array
    res = np.zeros((length_phrase, EMBED_DIM))  # glove dim

    # If more than one word
    if length_phrase > 1:

        # Get text, token, POS tags and phrase
        tokens = [elem[0] for elem in blob_phrase.tags]
        tags = [elem for elem in blob_phrase.tags]

        # Get words with part-of-speech of relevance
        phrase = [
            word
            for word, pos in tags
            if pos in ["NN", "NNP", "NNS", "NNPS", "VB", "JJ"]
        ]

    # If only one word
    else:
        phrase = [phrase.lower()]

    # For each token in phrase, get embedding
    for i, token in enumerate(phrase):
        if token in WORD2VEC:
            res[i] = WORD2VEC[token]

        # If not found, use lemmatized form
        else:
            lemmatised = Word(token).lemmatize()
            if lemmatised in WORD2VEC:
                res[i] = WORD2VEC[lemmatised]
            else:
                res[i] = np.zeros(EMBED_DIM)

    # Return embedding
    emb = np.mean(res, axis=0)
    return emb
