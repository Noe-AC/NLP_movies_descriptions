2022-03-10

# NLP on movies descriptions

On [Kaggle](https://www.kaggle.com) one can find some movies datasets:

1. Criterion movies dataset (e.g. [this one](https://www.kaggle.com/ikarus777/criterion-movies-collection)).
2. IMDB movies dataset.

Using Google's [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) algorithm to embed the movies descriptions in a vector space, it is possible to do various things among which:

1. Compute the pairwise distance between the descriptions of two movies. This can be used to find the *"closest"* movies to a given movie. For example, the five closest Criterion movies to [Fiend Without a Face](https://www.imdb.com/title/tt0050393/?ref_=fn_al_tt_1) (1958, A. Crabtree) are [Scanners](https://www.imdb.com/title/tt0081455/?ref_=nv_sr_srsg_0) (1981, D. Cronenberg), [Godzilla vs. Hedorah](https://www.imdb.com/title/tt0067148/?ref_=nv_sr_srsg_0) (1971, Y. Banno & I. Honda), [Genocide](https://www.imdb.com/title/tt0063195/?ref_=nv_sr_srsg_0) (1968, K. Nihonmatsu), [Godzilla vs. Gigan](https://www.imdb.com/title/tt0068371/?ref_=nv_sr_srsg_0) (1972, J. Fukuda & Y. Banno & I. Honda), [Night of the Living Dead](https://www.imdb.com/title/tt0063350/?ref_=fn_al_tt_1) (1968, G. A. Romero). This can be used to get inspired to watch movies.
2. Do a dimensional reduction (e.g. [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) or [UMAP](https://umap-learn.readthedocs.io/en/latest/)) of the embedding to do a 2d scatterplot of the movies.

Another thing that can be done is to embed a set of user queries in the same vector space.
Then the pairwise distance between the movies descriptions and the queries can be computed.
Therefore, for a given query, one can get the closest movies descriptions to that query.
For example, the closest Criterion movie to this query:

>"*Alien marketplace has a secret tunnel that leads to earth. Woman and her dog find it and arrive in alien marketplace. She is rescued by 3 naked humans who are kept as pets.*"

is the movie [The X from Outer Space](https://www.imdb.com/title/tt0062411/?ref_=nv_sr_srsg_0) (1967, K. Nihonmatsu) whose movie description reads:

>"*When a crew of scientists returns from Mars with a sample of the space spores that contaminated their ship, they inadvertently bring about a nightmarish earth invasion. After one of the spores is analyzed in a lab, it escapes, eventually growing into an enormous, rampaging beaked beast. An intergalactic monster movie from longtime Shochiku stable director Kazui Nihonmatsu,The X from Outer Spacewas the first in the studio’s short but memorable cycle of horror pictures.*"

Here I join a short Python script that does all that:

1. BERT Embedding of the movies descriptions and queries.
2. Dimensional reduction of the embedding.
3. Export the pairwise similarity matrix between movies to a CSV file.
4. Scatterplot of the dimensional reduction (however, not perfect because it is very dense).
5. Find closest movies to the user queries.
