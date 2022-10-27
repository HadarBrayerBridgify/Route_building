import re
import sys
import json
import uuid
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import Haversine as hs
from itertools import combinations
import data_preprocessing as dp
from typing import List, Dict, Any
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer, util




def model_embedding(df, col):
    """
    calculates the embeddings (as torch) of each entry in 'text' column according to SentenceTransformers

    Args:
      df: preprocessed DataFrame
      col: str, the name of the text column according to which the embeddings will be calculated

    Returns:
      tourch.Tensor
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("model:", type(model))

    # Single list of sentences
    sentences = df[col].values
    print("sentences:", type(sentences))
    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)  # each text transforms to a vector
    print("embedd:", type(embeddings))
    print("finished embeddings")
    return embeddings





def pairs_df_model(embeddings: Tensor, similarity_threshold: float) -> DataFrame:
    """
    Compute cosine-similarities of each embedded vector with each other embedded vector

    Args:
      embeddings: Tensor embeddings of the text column
      similarity_threshold: float, similarity score

    Returns:
      DataFrame with columns: 'ind1' (vector index), 'ind2' (vector index), 'score' (cosine score of the vectors)
      (The shape of the DataFrame is: rows: (n!/(n-k)!k!), for k items out of n)

    """

    cosine_scores: Tensor = util.cos_sim(embeddings, embeddings)
    pairs: List[Dict[str, Any]] = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            score = cosine_scores[i][j]
            if score >= similarity_threshold:
                pairs.append({"index": [i, j], "ind1": i, "ind2": j, 'score': np.float32(cosine_scores[i][j].item())})
    pairs_df = pd.DataFrame(pairs)

    pairs_df["ind1"] = pairs_df["ind1"].astype('uint16')
    pairs_df["ind2"] = pairs_df["ind2"].astype('uint16')

    return pairs_df


# def similarity_matrix(similarity_idx_df, reduced_df):
#     """
#     creates n^2 similarity matrix. Each attarction has a similarity score in relation to each attraction in the data
#
#     Args:
#       similarity_idx_df: DataFrame output of the function pairs_df_model
#       reduced_df: preprocessed DataFrame
#
#     Returns:
#       sqaure DataFrame. columns = index = the indices of the attractions. values: simialrity score
#     """
#     similarity_matrix = pd.DataFrame(columns=[i for i in range(reduced_df.shape[0])], index=range(reduced_df.shape[0]))
#     for i in range(reduced_df.shape[0]):
#         for j in range(i, reduced_df.shape[0]):
#             if j == i:
#                 similarity_matrix.iloc[i][j] = 1
#                 similarity_matrix.iloc[j][i] = 1
#             else:
#                 similarity_score = \
#                 similarity_idx_df[(similarity_idx_df["ind1"] == i) & (similarity_idx_df["ind2"] == j)]["score"].values
#                 similarity_matrix.iloc[i][j] = similarity_score
#                 similarity_matrix.iloc[j][i] = similarity_score
#     return similarity_matrix
#
#
# def change_idx_and_cols(similarity_matrix, df, col):
#     """
#     transform the name of the columns and indices to the name of the specified column
#
#     Args:
#       similarity_matrix: sqaure pd.DataFrame of similarity score of each attraction with each attraction
#       df: pd.DataFrame of the attractions
#       col: The name of the column according to which the columns will be named
#
#     Return:
#       list of dictionaries of similarity scores
#       """
#     similarity_matrix[col] = df[col]
#     similarity_matrix = similarity_matrix.set_index(col)
#     similarity_matrix.columns = similarity_matrix.index
#
#     return similarity_matrix.to_dict('records')


def groups_idx(similarity_df):
    """
    Creates a list of tuples, each tuple is a similarity group which contains the attractions indices (A group consists of the pairs of a particular index and the pairs of
    its pairs. There is no overlap of indices between the groups

    Args:
      similarity_df: DataFrame output of the function pairs_df_model

    Returns:
      a list of tuples. Each tuple contains attractions indices and represent a similarity group
    """
    sets_list = list()

    # go over all the index pairs in the dataframe
    for idx in similarity_df["index"].values:
        was_selected = False

        # list that contains all the groups sets
        first_match = set()

        for group in sets_list:
            # if idx has intersection with one of the groups, add the index to the group
            intersec = set(idx) & group
            if len(intersec) > 0:
                # add the index to the group
                group.update(idx)

                # save in the first group match (and collect if there are more matches)
                first_match.update(group)

                # remove the group (it will be inserted with all the matched items )
                sets_list.remove(group)

                # mark that we have intersection for not adding the idx as different group
                was_selected = True
        # after we iterate over all the groups and found the matches for the idx, insert first_match to the sets_list
        if len(first_match) > 0:
            sets_list.append(first_match)

        if not was_selected:
            sets_list.append(set(idx))
    return sets_list


def groups_df(similarity_df_above_threshold, df):
    """
    Creates a DataFrame of 'uuid' and 'similarity_uuid' of the attractions which have similarity score above the threshold

    Args:
      similarity_df_above_threshold: a filtered DataFrame of the output of pairs_df which pass 'score' > threshold
      df: pre-processed DataFrame of the attractions

    Returns:
      a DataFrame of 'uuid' and 'similarity_uuid'
    """

    # add 'group' column to the above threshold indices and order the dataframe by group
    display_columns = ['uuid']

    # extract the indices
    above_threshold_idx = list(set(np.array([idx for idx in similarity_df_above_threshold["index"]]).ravel()))
    print("above threshold..:", above_threshold_idx)

    # extract the relevant rows from the dataframe
    df_above_threshold = df.loc[above_threshold_idx][display_columns]
    df_above_threshold.columns = ["id"]
    df_above_threshold["id"] = df_above_threshold["id"].apply(lambda uuid: str(uuid))

    df_above_threshold['similarity_uuid'] = np.nan

    # divide the indices to groups according to similarity
    groups_list = groups_idx(similarity_df_above_threshold)
    # dropping groups above 50 attractions
    groups_list = [i for i in groups_list if len(i) < 50]

    # update the group columns according to the groups
    for group in groups_list:
        df_above_threshold['similarity_uuid'].loc[list(group)] = str(uuid.uuid4())
    df_above_threshold = df_above_threshold.dropna(axis="rows")
    similarity_groups_json = df_above_threshold.to_dict('records')
    return similarity_groups_json


def create_long_lat(df):
    df["geolocation"].fillna(0, inplace=True)
    df["long_lat"] = None
    empty_idx = df[df["geolocation"] == 0].index
    print(empty_idx)
    df["long_lat"].loc[empty_idx] = 0
    true_idx = set(df.index) - set(empty_idx)
    print(true_idx)
    df["long_lat"].loc[true_idx] = df["geolocation"].loc[true_idx].apply(
        lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][-2:])
    print("created 'lon_lat' col")


def create_distance_matrix(df_reduced):
    create_long_lat(df_reduced)
    distances_matrix = pd.DataFrame(columns=[i for i in range(df_reduced.shape[0])], index=range(df_reduced.shape[0]))
    for i in range(df_reduced.shape[0]):
        print(i)
        for j in range(i, df_reduced.shape[0]):
            loc1 = df_reduced["long_lat"][i]
            loc2 = df_reduced["long_lat"][j]
            if loc1 == 0 or loc2 == 0:
                dist_score = np.nan

            if len(loc1) < 2 or len(loc2) < 2:
                dist_score = np.nan
            else:
                dist_score = hs.haversine(loc1, loc2)  # distance in km

            distances_matrix.iloc[i][j] = dist_score
            distances_matrix.iloc[j][i] = dist_score


    return distances_matrix


def add_distance(similarity_scores, distance_matrix):
    similarity_scores["distance"] = None
    for i in range(similarity_scores.shape[0]):
        similarity_scores["distance"].iloc[i] = distance_matrix.iloc[similarity_scores["ind1"].iloc[i]][
            similarity_scores["ind2"].iloc[i]]


def add_title(df, similarity_score):
    similarity_score["title1"] = None
    similarity_score["title2"] = None
    for i in range(similarity_score.shape[0]):
        similarity_score["title1"].iloc[i] = df["title"].iloc[similarity_score["ind1"].iloc[i]]
        similarity_score["title2"].iloc[i] = df["title"].iloc[similarity_score["ind2"].iloc[i]]


def split_large_group(group_to_split: pd.DataFrame, unmatch_df):
    groups_list = list()
    print("group_idx:", group_to_split.index)
    print("len group:", group_to_split.shape[0])
    for idx in group_to_split.index:
        idx_break = 0

        for group in groups_list:
            group_break = 0
            for number in group:
                if unmatch_df[((unmatch_df["ind1"] == idx) & (unmatch_df["ind2"] == number)) | (
                        (unmatch_df["ind1"] == number) & (unmatch_df["ind2"] == idx))].shape[0]:
                    group_break = 1
                    break
            if not group_break:
                group.append(idx)
                idx_break = 1
                break
        if not idx_break:
            groups_list.append([idx])
    return groups_list


def extract_group_scores(group_df, similarity_scores):
    scores_df = pd.DataFrame()
    for idx_couple in combinations(group_df.index, 2):
        idx_row = similarity_scores[
            ((similarity_scores["ind1"] == idx_couple[0]) & (similarity_scores["ind2"] == idx_couple[1])) | (
                        (similarity_scores["ind1"] == idx_couple[1]) & (similarity_scores["ind2"] == idx_couple[0]))]
        scores_df = pd.concat([scores_df, idx_row])
    return scores_df


def main(data: List[Dict]):
    raw_df = pd.DataFrame.from_dict(data)

    df_processed = dp.data_preprocess(raw_df)
    df_processed = df_processed[df_processed["title"] != ""]
    df_processed.reset_index(drop=True, inplace=True)

    # Creating similarity DataFrame according to 'text'
    embeddings_text = model_embedding(df_processed, "title")
    embeddings = pd.DataFrame(embeddings_text)
    print("shape embeddings:", embeddings.shape)
    similarity_df = pairs_df_model(embeddings_text, 0.6)

    #similarity_df = similarity_df[similarity_df["score"] >= 0.6]
    distance_matrix = create_distance_matrix(df_processed)
    add_distance(similarity_df, distance_matrix)
    add_title(df_processed, similarity_df)
    print(similarity_df.shape)

    # filtering according to 'title' column.
    max_similarity_threshold = 0.78
    min_similarity_threshold = 0.6
    distance_threshold = 0.1
    similarity_df_threshold = similarity_df[similarity_df["score"] > max_similarity_threshold]
    similarity_dist_df_threshold = similarity_df[
        (similarity_df["score"] > min_similarity_threshold) & (similarity_df["distance"] < distance_threshold)]
    similarity_df_above_threshold = pd.concat([similarity_df_threshold, similarity_dist_df_threshold])
    # similarity_df_above_threshold = similarity_df_threshold.copy()
    similarity_df_above_threshold.drop_duplicates(subset=similarity_df_above_threshold.columns[1:], inplace=True)

    similarity_df_json = groups_df(similarity_df_above_threshold, df_processed)

    return similarity_df_json, similarity_df


if __name__ == "__main__":
    #distance_matrix = pd.read_csv("ny")
    data = pd.read_csv("new_york_attractions.csv")
    data_json = data.to_dict('records')
    same_attraction_tickets_json, similarity_scores_df = main(data_json)
    same_attraction_tickets_df = pd.DataFrame.from_dict(same_attraction_tickets_json)
    same_attraction_tickets_df.to_csv("same_attraction_tickets_ny.csv")