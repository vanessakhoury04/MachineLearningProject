# I. Project Objective and Data Description

This project aims to make book recommendations to users based on their preferences. To do so, we build a model that gives the most accurate recommendations.

The data used to train the model is provided in `user_preferences.csv`. This data serves to build a binary interaction matrix indicating which books each user has read.

The `items.csv` matrix is used to improve the model by drawing similarity between books based on their metadata. This yields better results in some cases, including the final recommender model.

In some trials, the data is also augmented using the Google Books API, but this is not retained for the final model as explained in later sections.

Most of the methods applied in building this model are based on the class lectures and lab session code. Some other methods such as the decay function and sentence-transform embeddings are attempted based on more extended research assisted by AI tools.

---

# II. Best Model Description

The best model is a combination between a user-user, item-item, and content-based recommender system. They are weighted `0.3`, `0.3`, and `0.4` respectively in the final recommendation.

This model first creates an interaction matrix from the `interactions_train.csv` provided in the data files. It weights each interaction with an exponential decay function that gives the highest weight to the most recent entries as compared to the most recent one across the entire dataset (Section III.6).

Then, the code cleans the metadata for each book as given in `items.csv`, and creates a content matrix using the title, author, and subject of each book with a TF-IDF approach (Section III.4).

Once the data is prepared, the code calculates the cosine similarity for the:
- user-user matrix
- item-item matrix
- content matrix

and computes the likelihood that the user will interact with an item based on the following formulas provided in the recommender lab:

## Item-item collaborative filtering

$$
p_u(i)=\frac{\sum_{i'} sim(i,i') \cdot R_u(i')}{\sum_{i'} sim(i,i')}
$$

## User-user collaborative filtering

$$
p_u(i)=\frac{\sum_{u'} sim(u,u') \cdot R_{u'}(i)}{\sum_{u'} sim(u,u')}
$$

Where:

- $p_u(i)$ is the likelihood of user $u$ interacting with item $i$
- $sim(i,i')$ is the cosine similarity between items $i$ and $i'$
- $sim(u,u')$ is the cosine similarity between users $u$ and $u'$
- $R_u(i')$ is 1 if user $u$ has already interacted with item $i'$, and otherwise 0
- $R_{u'}(i)$ is 1 if user $u'$ has already interacted with item $i$, and otherwise 0

Finally, the code gives a final prediction combining all three models, weighted:
- `0.3` for user-user
- `0.3` for item-item
- `0.4` for content-based

This is the weight combination that gives the best score on the Kaggle leaderboard: **0.1655**.

Sample recommendations based on this model can be collected on our website designed specifically to run this recommender.

Many other approaches were considered while building this model. The following section details the most relevant ones, as well as the precision and recall of the results.

---

# III. Alternative Models

## 1. Summary of Results

The following table presents a summary of the precision and recall for several attempted models. The components and operation of each model are further detailed in the rest of the section.

| Model | Precision @ k=10 | Recall @ k=10 |
|---|---|---|
| User-user (R00) | 0.05653 | 0.29065 |
| Item-item (R00) | 0.05561 | 0.26399 |
| Hybrid without content (R01) | 0.06082 | 0.29224 |
| Hybrid with content (R01-with) | 0.06142 | 0.29726 |
| Nearest neighbor k=150 (R07) without content (0.55, 0.45) | 0.06078 | 0.29213 |
| Nearest neighbor k=150 (R07) with content (0.55, 0.45) | 0.06110 | 0.29482 |
| Decay (R08-decay) - Final | 0.06137 | 0.29596 |
| Embedding | 0.06095 | 0.29336 |

> Rerun nearest neighbor as a save-as of hybrid with content to check if results for neighbor are accurate.

---

## 2. User-user and Item-item Models

These two models are the basis of the final code. They are built separately using the same probability formulas presented above, as per the recommender lab.

If run on the full data, the user-user model results in the baseline score of `0.1452` on the leaderboard.

---

## 3. Hybrid Model Without Metadata

Once a first simple model was built, the immediate idea was to combine them.

This model provides recommendations based on both:
- the user-user model
- the item-item model

For a weight of `0.55` for the user-user model, it provides the score of `0.1643` on the leaderboard.

---

## 4. Hybrid Model With Metadata

Another layer of complexity was added to the model with the metadata in the `items.csv` file.

The data included in this model is:
- title
- author
- subject

The subjects are cleaned into distinct words, and punctuation as well as stop words are removed.

All three fields are then combined into one single content field, which is converted into vectors using TF-IDF with the following arguments:

- `max_features = 10000`
- `ngram_range = (1, 3)`
- `min_df = 3`
- `max_df = 0.7`

The precision and recall of this model are superior to the one without content for a combination of weights where the user-user model is `0.5`.

However, the score on the Kaggle leaderboard remains between `0.16` and `0.1638` depending on the weights, which is not an improvement over the previous model.

This content model based on metadata is implemented in the final code nonetheless, as its precision and recall are superior, and the final model performs better with it than without it on the leaderboard.

The `items.csv` metadata was also augmented using the Google Books API. This provided:
- descriptions of certain books
- categories for most books

However, other data such as rankings could not be obtained, and the model did not yield a better result than the non-augmented metadata.

For this reason, only non-augmented metadata is used in the final model.

---

## 5. Nearest Neighbor

In the interest of improving the user-user model, we attempt to find a similarity between user \(u\) and only \(k\) of its nearest neighbors.

A series of trials performed to obtain the optimal number of neighbors for precision and recall yields:

\[
k = 150
\]

However, adding this to the model does not improve the precision and recall compared to the hybrid model.

We assume that:
- a too-small \(k\), such as `25` or `50`, deprives the model of relevant data
- hence producing lower precision and recall than larger values such as `150`

But \(k=150\) did not yield a better result because neighbors beyond 150 might not have been relevant enough to improve performance even if included.

For this reason, the final model does not include the k-neighbor approach.

---

## 6. Decay

The `interactions_train.csv` file provides information on when a book was read by a specific user.

We use this information to build a decay function that weights the interaction matrix based on the following formula:

$$
w = e^{-\lambda (t_{\max} - t)}
$$

Where:

- $\lambda$ is the optimized decay rate
- $t_{\max}$ is the most recent timestamp in the dataset
- $t$ is the time at which user $u$ read item $i$

This approach combined with the non-augmented metadata hybrid model results in the best leaderboard score:

$$
0.1655
$$

---

## 7. Embedding

In an attempt to improve the metadata matrix of the final model, we use SentenceTransformer embeddings instead of the TF-IDF method.

Sentence Transformers generate dense semantic embeddings of the metadata, enabling the recommender system to capture contextual similarity between books beyond the simple keyword matching of TF-IDF.

The model yields a maximum leaderboard score of `0.1643`, for weights shifted more towards the content matrix, with the following distribution:

- `0.2` user-user
- `0.2` item-item
- `0.6` content matrix

all other elements of the final code included as described in Section II.

This is not better than the code without embeddings. For this reason, the final recommender model operates on TF-IDF.

---

## 8. Discussion of Results

The final code builds recommendations based on a hybrid model of:
- user-user collaborative filtering
- item-item collaborative filtering
- content-based filtering using TF-IDF metadata

Enhancements of this hybrid model were made using a decay function to give more importance to books that were read more recently.

Fine-tuning parameters such as:
- TF-IDF arguments
- weight distributions

allowed further optimization of the overall performance.

As a result of the other attempted approaches, we also conclude that the interaction data is more informative than semantic content features.

This is supported by the fact that:
- embeddings
- augmented metadata

decreased the performance of the model.

---

# IV. User Interface

The user interface is designed using Streamlit.

First, it allows users to obtain recommendations based on `user_preferences.csv`. In other words, the users in the provided dataset act as registered users of the website who can immediately obtain recommendations.

The user interface also provides an option for new users to obtain recommendations.

This option is based on the same recommender, called as a function to compute the similarity of the new user to the already-computed similarity matrices.

The new user can enter as many books as they like.

However, because there is no field allowing the user to indicate when they read the books (as this is not usual information to request from users), this onboarding model does not use the decay function for the new user, while still maintaining this feature in the pretrained recommendation model.