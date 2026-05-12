#Lab code version of the project

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
# Import for text analytics
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim import corpora
import multiprocessing
import re
from sklearn.feature_extraction.text import TfidfVectorizer

items=pd.read_csv("../Data/items.csv")
user_pref=pd.read_csv("../Data/interactions_train.csv")


#EDA
n_users = user_pref.u.nunique()
n_items = len(items)
print(f'Number of users = {n_users}, \n Number of movies = {n_items} \n Number of interactions = {len(user_pref)}')

#splitting into train-test
user_pref=user_pref.sort_values(["u", "t"])
user_pref["pct_rank"] = user_pref.groupby("u")["t"].rank(pct=True, method='dense')
user_pref.reset_index(inplace=True, drop=True)

train_data = user_pref[user_pref["pct_rank"] < 0.8]
test_data = user_pref[user_pref["pct_rank"] >= 0.8]

# Define a function to create the data matrix
def create_data_matrix(data, n_users, n_items):
    """
    This function returns a numpy matrix with shape (n_users, n_items).
    Each entry is a binary value indicating positive interaction.
    """
    data_matrix = np.zeros((n_users, n_items))
    print(data["i"].max(), n_items)
    print(data_matrix.shape)
    data_matrix[data["u"].values, data["i"].values] = 1
    return data_matrix

def create_decayed_matrix(data, n_users, n_items, decay_rate=1e-9):
    data_copy=data.copy()
    data_matrix = np.zeros((n_users, n_items))
    
    max_t = data_copy["t"].max()
    
    # Calculate weight: e^(-lambda * delta_t)
    # delta_t is the time difference (e.g., in days or hours)
    data_copy["weight"] = np.exp(-decay_rate * (max_t - data_copy["t"]))
    
    data_matrix[data_copy["u"].values, data_copy["i"].values] = data_copy["weight"].values
    return data_matrix


# Create the training and testing matrices
train_data_matrix = create_decayed_matrix(train_data, n_users, n_items)
test_data_matrix = create_data_matrix(test_data, n_users, n_items)

#create metadata vectors
sp = spacy.load('fr_core_news_sm')
def remove_duplicates(tokens):
    seen = set()
    result = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result

def spacy_tokenizer(text):

    # Define stopwords, punctuation, and numbers
    stop_words = spacy.lang.fr.stop_words.STOP_WORDS
    punctuations = string.punctuation.replace("-", "") + '–—'
    numbers = "0123456789"
    
    if not isinstance(text, str):
        return ""
    
    # 1. Replace separators properly (regex)
    text = re.sub(r"[;,\/]+", " ", text)

    # 2. Split into words
    mytokens = text.split()
    
    # Lemmatize each token and convert each token into lowercase
    mytokens = ([ word.lower().strip() for word in mytokens ])

    # Remove stop words and punctuation
    mytokens = ([ word for word in mytokens 
                 if word not in stop_words and word not in punctuations ])
    
   
    mytokens_2 = []
    for word in mytokens:
        # This effectively "cleans" the word character by character
        clean_word = "".join([char for char in word if char not in punctuations and char not in numbers])
        
        if clean_word != "":
            mytokens_2.append(clean_word)

    # Return preprocessed list of tokens
    mytokens_2 = remove_duplicates(mytokens_2)
    return " ".join(mytokens_2)


items["titles_cleaned"]=[title[:-2] for title in items["Title"]]
items["subjects_cleaned"] = [spacy_tokenizer(text) for text in items["Subjects"]]
items["authors_cleaned"] = [spacy_tokenizer(text) for text in items["Author"]]
items["concepts"]=items["titles_cleaned"] + " " + items["authors_cleaned"] + " " + items["subjects_cleaned"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3),
    min_df=3,
    max_df=0.7,
    sublinear_tf=True
)
content_matrix = vectorizer.fit_transform(items["concepts"])

content_sim = cosine_similarity(content_matrix)
def content_based_predict(interactions, similarity, epsilon=1e-9):
    """
    interactions: your train_data_matrix (Users x Items)
    similarity: the content_sim matrix (Items x Items)
    """
    # Matrix math: (Users x Items) dot (Items x Items) = (Users x Items)
    # This sums up the content-similarity scores for all books a user has read.
    pred = interactions.dot(similarity) / (np.abs(similarity).sum(axis=1) + epsilon)
    return pred

content_prediction = content_based_predict(train_data_matrix, content_sim)

#item-item matrix

item_similarity = cosine_similarity(train_data_matrix.T)


# Define the function to predict interactions based on item similarity
def item_based_predict(interactions, similarity, epsilon=1e-9):
    """
    Predicts user-item interactions based on item-item similarity.
    Parameters:
        interactions (numpy array): The user-item interaction matrix.
        similarity (numpy array): The item-item similarity matrix.
        epsilon (float): Small constant added to the denominator to avoid division by zero.
    Returns:
        numpy array: The predicted interaction scores for each user-item pair.
    """
    # np.dot does the matrix multiplication. Here we are calculating the
    # weighted sum of interactions based on item similarity
    pred = similarity.dot(interactions.T) / (similarity.sum(axis=1)[:, np.newaxis] + epsilon)
    return pred.T  # Transpose to get users as rows and items as columns

# Calculate the item-based predictions for positive interactions
item_prediction = item_based_predict(train_data_matrix, item_similarity)
print("Predicted Interaction Matrix:")
print(item_prediction)
print(item_prediction.shape)


#user-user matrix
user_similarity = cosine_similarity(train_data_matrix)



# Define the function to predict interactions based on user similarity
def user_based_predict(interactions, similarity, epsilon=1e-9):
    """
    Predicts user-item interactions based on user-user similarity.
    Parameters:
        interactions (numpy array): The user-item interaction matrix.
        similarity (numpy array): The user-user similarity matrix.
        epsilon (float): Small constant added to the denominator to avoid division by zero.
    Returns:
        numpy array: The predicted interaction scores for each user-item pair.
    """
    # Calculate the weighted sum of interactions based on user similarity
    pred = similarity.dot(interactions) / (np.abs(similarity).sum(axis=1)[:, np.newaxis] + epsilon)
    return pred

# Calculate the user-based predictions for positive interactions
user_prediction = user_based_predict(train_data_matrix, user_similarity)
print("Predicted Interaction Matrix (User-Based):")
print(user_prediction)
print(user_prediction.shape)



def precision_recall_at_k(prediction, ground_truth, k=10):
    """
    Calculates Precision@K and Recall@K for top-K recommendations.
    Parameters:
        prediction (numpy array): The predicted interaction matrix with scores.
        ground_truth (numpy array): The ground truth interaction matrix (binary).
        k (int): Number of top recommendations to consider.
    Returns:
        precision_at_k (float): The average precision@K over all users.
        recall_at_k (float): The average recall@K over all users.
    """
    num_users = prediction.shape[0]
    precision_at_k, recall_at_k = 0, 0

    for user in range(num_users):
        # TODO: Get the indices of the top-K items for the user based on predicted scores
        top_k_items = np.argsort(prediction[user, :])[-k:]

        # TODO: Calculate the number of relevant items in the top-K items for the user
        relevant_items_in_top_k = np.isin(top_k_items, np.where(ground_truth[user, :] == 1)[0]).sum()

        # TODO: Calculate the total number of relevant items for the user
        total_relevant_items = ground_truth[user, :].sum()

        # Precision@K and Recall@K for this user
        precision_at_k += relevant_items_in_top_k / k
        recall_at_k += relevant_items_in_top_k / total_relevant_items if total_relevant_items > 0 else 0

    # Average Precision@K and Recall@K over all users
    precision_at_k /= num_users
    recall_at_k /= num_users

    return precision_at_k, recall_at_k


precision_user_k, recall_user_k = precision_recall_at_k(user_prediction, test_data_matrix, k=10)
precision_item_k, recall_item_k = precision_recall_at_k(item_prediction, test_data_matrix, k=10)

print('User-based CF Precision@K:', precision_user_k)
print('User-based CF Recall@K:', recall_user_k)
print('Item-based CF Precision@K:', precision_item_k)
print('Item-based CF Recall@K:', recall_item_k)


#make predictions
alpha = 0.3
beta=0.3
top_k = 10

# Combine predictions once
final_prediction = alpha * user_prediction + (beta) * item_prediction + (1 - alpha - beta)*content_prediction
precision_hybrid_k, recall_hybrid_k = precision_recall_at_k(final_prediction, test_data_matrix, k=10)
print('Hybrid-based CF Precision@K:', precision_hybrid_k)
print('Hybrid-based CF Recall@K:', recall_hybrid_k)

rows = []

for x in user_pref["u"].unique():
    top_10 = np.argsort(final_prediction[x, :])[-10:][::-1]
    user_str = " ".join(map(str, top_10))
    rows.append((x, user_str))
    

# Final dataframe
hybrid_df = pd.DataFrame(rows, columns=["user_id", "recommendation"])
hybrid_df.to_csv("../Submission/Hybrid_0.3_0.3_SBB_R08_final.csv", index=False)





