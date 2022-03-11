2022-03-10

# NLP on movies descriptions

On [Kaggle](https://www.kaggle.com) one can find some movies datasets:

1. Criterion movies dataset (e.g. [this one](https://www.kaggle.com/ikarus777/criterion-movies-collection)).
2. IMDB movies dataset.

Using Google's BERT algorithm to embed the movies descriptions in a vector space, it is possible to do various things among which:

1. Compute the pairwise distance between the descriptions of two movies. This can be used to find the *closest* movies to a given movie. For example, the closest Criterion movies to **Fiend Without a Face** are **Scanners**, **Godzilla vs. Hedorah**, **Genocide**, **Godzilla vs. Gigan**, **Night of the Living Dead**, and so on. This can be used to get inspired to watch movies.
2. Do a dimensional reduction (e.g. [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) or [UMAP](https://umap-learn.readthedocs.io/en/latest/)) of the embedding to do a 2d scatterplot of the movies.

Another thing that can be done is to embed a set of user queries in the same vector space.
Then the pairwise distance between the movies descriptions and the queries can be computed.
Therefore, for a given query, one can get the closest movies descriptions to that query.
For example, the closest Criterion movie to this query:

>"*Alien marketplace has a secret tunnel that leads to earth. Woman and her dog find it and arrive in alien marketplace. She is rescued by 3 naked humans who are kept as pets.*"

is the movie **The X from Outer Space** whose movie description reads:

>"*When a crew of scientists returns from Mars with a sample of the space spores that contaminated their ship, they inadvertently bring about a nightmarish earth invasion. After one of the spores is analyzed in a lab, it escapes, eventually growing into an enormous, rampaging beaked beast. An intergalactic monster movie from longtime Shochiku stable director Kazui Nihonmatsu,The X from Outer Spacewas the first in the studio’s short but memorable cycle of horror pictures.*"

Here I join a short Python script that does all that:

1. BERT Embedding of the movies descriptions and queries.
2. Dimensional reduction of the embedding.
3. Export the pairwise similarity matrix between movies to a CSV file.
4. Scatterplot of the dimensional reduction (however, not perfect because it is very dense).
5. Find closest movies to the user queries.
