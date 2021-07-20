# skills_taxonomy

### In a Nutshell

We want to build a skills taxonomy prototpye. So we start by **loading and inspecting the provided data** and its structure. We build several **helper functions, classes and dictionaries and graphs** for easier processing at later stages. We also **visualize the ISCO structure** and its associated occupations to gain a better understanding.

Using the generated circular tree ISCO graph and the information about skills, occupations and ISCO ID, we develop a **measure for calculating the relevance of a skill** at any given node in the ISCO graph, by adapting the _tf-idf_ measure. The _tf-idf_ represents the relevance of a skill with regards to the skills frequency and distribution across the graph.

The _tf-idf_ for skill at each node in the ISCO graph is used as a feature set. In order to fight the curse of dimensionality, we apply **dimensinality reduction using latent semantic analysis (LSA)**. We employ **k-means clustering** for finding groups of skills with similar features.

Finally, for each skill in a cluster, we compute the two most similar skills by **computing the cosine similarity of the word embeddings** of their skill description. We **rank the skills by their _document frequency_** (collected at the feature engineering step), which allows the **creation of candidate branches in the taxonomy**.

### Structure

The code is refactored into seperate modules. To see an application of the various functions, check out the notebook [Build_Skills_Taxonomy.ipynb](https://github.com/athrado/skills_taxonomy/blob/0_setup_cookiecutter/skills_taxonomy/analysis/notebooks/Build_Skills_Taxonomy.ipynb) in the folder `skills_taxonomy/analysis/notebooks`.

The notebook shows a prototype for solving **building a skills taxonomy**. It describes the entire process for developing this approach, including data analysis and some processing steps that are not necessary for the final pipeline.

- **Loading, Analysing and Preprocessing**

  - Load the data
  - Inspect the data
  - Preprocess and build dictionaries
  - Class for ISCO IDs
  - Get essential and option skills
  - Get skills for occupaton top group

- **Visualization**

  - First visualization of occupations with ISCO similarity
  - Reduced ISCO graph for understanding ISCO structure
  - Full directed ISCO graph for futher computation

- **Feature Engineering**

  - Compute number leaf nodes for given node
  - Compute term frequency
  - Compute document frequency
  - Compute tf-idf feature array

- **Clustering**

  - Reduce feature set
  - Scale features (MinMaxScaler)
  - Dimensionality reduction using LSA
  - k-means clustering
  - Load skills data and inspect clusters

- **Taxonomy**

  - Embed skill descriptions using word embeddings
  - Compute cosine similarity
  - Find most similar skills by description
  - Use document frequency for ranking skills
  - Build taxonomy branches
  - Inspect results

- **Conclusion**

### Setup

**Before getting started**, meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:

- Install: `git-crypt` and `conda`
- Have a Nesta AWS account configured with `awscli`

Then use the following commands:

``` 
# Get the repository
$ git clone https://github.com/athrado/skills_taxonomy.git

# Setup
$ cd skills_taxonomy
$ make install
$ conda activate skills_taxnonomy

# Prepare input data
$ make inputs-pull
$ unzip "inputs/data/word2vec.zip" -d inputs/data/.

# Setup textblob
$ python -m textblob.download_corpora

# Setup ipygraphviz (or check here: https://pygraphviz.github.io/)
brew install graphviz
conda install -c conda-forge pygraphviz
```
