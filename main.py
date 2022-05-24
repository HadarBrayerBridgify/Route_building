import os
import re
import sys
import logging
import warnings
import argparse
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from configparser import ConfigParser


class Anchors:

    def __init__(self, anchors):
        self.anchors = dict(sorted(anchors.items(), key=lambda x: x[1]))
        self.keys_list = list(self.anchors.keys())
        self.position_list = list(self.anchors.values())
        self.num_anchors = len(anchors)

    def num_attrac_before(self):
        return min(self.position_list) - 1

    def num_attrac_after(self, tot_num_attrac):
        return tot_num_attrac - max(self.position_list)

    def num_attrac_between(self, position1, position2):
        return abs(position2 - position1) - 1


class RouteBuilder:

    def __init__(self, df, chosen_tags, popularity_vec, df_similarity_norm, df_distances_norm, tot_num_attractions,
                 weight_dict, anchors=False):
        self.df = df
        # self.tags_vec = tags_vec
        self.chosen_tags = chosen_tags
        self.popularity_vec = popularity_vec
        self.df_similarity_norm = df_similarity_norm
        self.df_distances_norm = df_distances_norm
        self.tot_num_attractions = tot_num_attractions
        self.weight_dict = weight_dict
        self.anchors = anchors

        self.chosen_idx = list()  # np.nan(self.tot_num_attractions)
        self.similarity_vec = 0
        self.distance_vec = None
        self.tags_vec = None
        if len(self.chosen_idx) > 0:
            self.last_idx = self.chosen_idx[-1]

    def first_attraction(self):
        """
        return the first attraction according to 2*popularity and chosen tags
        """

        self.chosen_idx = [(self.tags_vec + 2 * self.popularity_vec).sort_values().index[0]]

    def attractions_score_vec(self, idx_to_drop):
        """
        return a vector with a score for every attraction according to the last chosen attraction/s
        """
        vectors_results = (self.weight_dict["popular"] * self.popularity_vec) + (
                    self.weight_dict["distance"] * self.distance_vec) + (
                                      self.weight_dict["similarity"] * self.similarity_vec) + (
                                      self.weight_dict["tags"] * self.tags_vec)

        # drop the chosen indices
        vectors_results.drop(index=idx_to_drop, inplace=True)

        return vectors_results

    def next_best(self, score_vector):
        """
        input: Vector with the score of each attraction according to the last chosen attraction
        output: the next best attraction/s
        """
        return score_vector.sort_values().index[0]

    def df_tags(self):
        """
        Creating DataFrame for all tags (each tag will have a different column)
        """

        tags_dict = {tag: [] for tag in self.chosen_tags}
        for tag_name in tags_dict.keys():
            for tags in self.df["prediction"]:
                if tag_name in tags:
                    tags_dict[tag_name].append(1)
                else:
                    tags_dict[tag_name].append(0)
        return pd.DataFrame(tags_dict)

    def update_tags_vec(self):

        # creating a dataframe for the chosen tags
        tags_df = self.df_tags()

        # reduce the values of the tags that already been chosen
        if len(self.chosen_idx) > 0:

            # checking how many times each 'chosen tag' has been chosen
            selected_tags_count = tags_df.iloc[self.chosen_idx].sum()

            # Consider only those above 0
            selected_tags_count = selected_tags_count[selected_tags_count > 0]

            for i in range(selected_tags_count.shape[0]):
                tag_name = selected_tags_count.index[i]
                tag_value = selected_tags_count.values[i]
                tags_df[tag_name] = tags_df[tag_name] * (1 / (2 * tag_value))

        # sum the tags to a vector
        tags_sum = tags_df.sum(axis=1)

        # normalized the vector to be between 0-1. transforming the results with '1-norm' in order for the best
        # attraction in terms of tags to be the lowest. Adding 0.01 in order not to reset the results
        self.tags_vec = 1 - self.norm_df(tags_sum) + 0.01

    def norm_df(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def current_similarity_vec(self):
        """
        Return the similarity vector associated with the last attraction along with the other selected attractions
        """
        current_similarity_vec = self.df_similarity_norm.iloc[self.chosen_idx[-1]]
        self.similarity_vec = current_similarity_vec + (1 / 3) * self.similarity_vec

    def update_distance_vec(self):
        self.distance_vec = self.df_distances_norm.iloc[self.chosen_idx[-1]]

    def attraction_between_anchors(self, vector1, vector2):
        """
        input: 2 vectors with the attraction score according to each anchor
        output: the best attraction idx between the anchors
        """
        return (vector1 + vector2).sort_values().index[0]

    def update_vectors(self):
        """
        update the vectors according to chosen_idx (according to the attractions that were chosen)
        """
        # update the tags_vector
        self.update_tags_vec()

        # find similarity_vec according to current attraction
        self.current_similarity_vec()

        # similarity_vec = current_similarity_vec(chosen_idx, df_similarity_norm)
        self.similarity_vec = self.norm_df(self.similarity_vec)

        # extract distance vector
        self.update_distance_vec()
        print("Successfully updated vectors!")

    def select_attractions_idx(self, num_attractions, idx_to_drop):  # idx_to_drop was added mainly for the anchors

        for i in range(num_attractions):
            self.update_vectors()

            # Select the next best attraction
            # print("idx_to drop:", idx_to_drop)
            vector_results = self.attractions_score_vec(idx_to_drop)
            attraction_idx = self.next_best(vector_results)

            # append the next index to the indices list
            self.chosen_idx.append(attraction_idx)
            print("chosen_idx select_attractions_idx:", self.chosen_idx)
            idx_to_drop.append(attraction_idx)

    def idx_to_df(self):
        # present the selected attractions by order:
        selected_attractions = self.df.iloc[self.chosen_idx].drop(columns=["index", "Unnamed: 0"])
        return selected_attractions

    def select_middle_attrac(self, anchors):
        """
         input anchors= {position: idx, position: idx}): {0:[10], 2:[40,30]}, the anchore always should be the last item in the list
         output: attraction idx between the anchors: (between 10 and 30)
        """
        ## select the attraction in the middle
        # reset the chosen_idx and update the vectors according to the first anchore and its chosen attractions
        idx_to_drop = anchors[0] + anchors[2]
        self.chosen_idx = anchors[0]
        self.similarity_vec = 0
        self.update_vectors()
        new_anchore1_vec = self.attractions_score_vec(idx_to_drop)

        # reset the chosen_idx and update the vectors according to the second anchore and its chosen attractions
        self.chosen_idx = anchors[2]
        self.similarity_vec = 0
        self.update_vectors()
        new_anchore2_vec = self.attractions_score_vec(idx_to_drop)

        # find the middle attraction according to both anchores
        middle_attraction_idx = self.attraction_between_anchors(new_anchore1_vec, new_anchore2_vec)
        # print("middle attraction", middle_attraction_idx)

        return middle_attraction_idx

    def even_attractions_between_anchors(self, idx_vec):  # np.array([0,10,20,30,40,50])
        """
        find the best 2 middle attractions according to both anchors (find the best "20, 30")
        """
        for i in range(5):
            print(f"itteration {i}")
            middle1_position = len(idx_vec) // 2 - 1  # 2
            middle2_position = len(idx_vec) // 2  # 3
            middle1 = idx_vec[middle1_position]  # 20
            middle2 = idx_vec[middle2_position]  # 30

            # find middle 1
            print("middle1:", middle1)
            middle1_new = self.select_middle_attrac(
                {0: idx_vec[:middle1_position], 2: idx_vec[middle2_position::][::-1]})
            print("middle1_new:", middle1_new)
            # update the new middle 1 in the list
            idx_vec[middle1_position] = middle1_new

            # find middle 2
            print("middle2:", middle2)
            middle2_new = self.select_middle_attrac(
                {0: idx_vec[:middle2_position], 2: idx_vec[middle2_position + 1::][::-1]})
            print("middle2:", middle2_new)
            # update the new middle 2 in the list
            idx_vec[middle2_position] = middle2_new

            if middle1 == middle1_new and middle2 == middle2_new:
                print("found the best middle attractions!")
                return idx_vec
        return idx_vec

    def select_attractions_between_anchors(self, anchors):
        # when we have an odd number of attractions between the
        # anchors. anchors example: {10:1, 50:5}

        # number of attractions to select according to each anchore
        num_attractions_per_anchore = anchors.num_attrac_between(anchors.position_list[0],
                                                                 anchors.position_list[1]) // 2
        print("num_attractions_per_anchore:", num_attractions_per_anchore)

        # first anchore:
        # insert the first anchore to chosen_idx
        self.chosen_idx = [anchors.keys_list[0]]
        # create a list with the anchors and their chosen attractions idx
        idx_to_drop = anchors.keys_list

        # selecting attraction/s according to the first anchore
        self.select_attractions_idx(num_attractions_per_anchore, idx_to_drop)
        first_anchore_idx = self.chosen_idx.copy()
        print("chosen_idx:", self.chosen_idx)  # [10, 20]
        print("idx to drop:", idx_to_drop)

        # add the second anchore to the list
        idx_to_drop.append(anchors.keys_list[1])  # [10, 20, 50]

        # redefine chosen_idx with the second anchore
        self.chosen_idx = [anchors.keys_list[1]]

        # reset self.similarity_vec (that we will not have the history value of the first anchore)
        self.similarity_vec = 0

        # adding attractions according to the second anchore, drop the attractions that were already chosen by the
        # first anchore
        self.select_attractions_idx(num_attractions_per_anchore, idx_to_drop)
        second_anchore_idx = self.chosen_idx.copy()
        print("chosen_idx:", self.chosen_idx)  # [50, 40]

        ### select the attraction in the middle
        # middle_attraction_idx = self.select_middle_attrac({0:first_anchore_idx, 2:second_anchore_idx})

        # join the indices to one list and reverse the second list so that the anchore will be the last item
        anchors_idx = first_anchore_idx + second_anchore_idx[::-1]  # [10, 20, 40, 50]
        print("anchors idx", anchors_idx)

        num_attractions_between = anchors.num_attrac_between(anchors.position_list[0], anchors.position_list[1])
        if num_attractions_between == 0:
            self.chosen_idx = anchors_idx

        elif num_attractions_between % 2 == 0:
            self.chosen_idx = self.even_attractions_between_anchors(anchors_idx)

        else:
            ### select the attraction in the middle
            middle_attraction_idx = self.select_middle_attrac({0: first_anchore_idx, 2: second_anchore_idx})

            # add the middle attraction to the idx list
            anchors_idx.insert(num_attractions_per_anchore + 1, middle_attraction_idx)
            self.chosen_idx = anchors_idx

        print("chosen attractions by both anchors", self.chosen_idx)

        # reset similarity_vec
        self.similarity_vec = 0

    def attractions_before_anchore(self, anchors):
        """
        select attractions idx before anchore
        """
        num_attrac_to_select = anchors.num_attrac_before()
        # Drop off the attractions that have already been selected
        idx_to_drop = self.chosen_idx
        # reverse chosen_idx in order to choose the "before attractions"
        self.chosen_idx = self.chosen_idx[::-1]
        self.select_attractions_idx(num_attrac_to_select, idx_to_drop)
        # reverse again chosen_idx
        self.chosen_idx = self.chosen_idx[::-1]

    def attractions_after_anchors(self, anchors):
        """
        select attractions idx after anchore
        """
        num_attrac_to_select = anchors.num_attrac_after(self.tot_num_attractions)
        # Drop off the attractions that have already been selected
        idx_to_drop = self.chosen_idx.copy()
        self.select_attractions_idx(num_attrac_to_select, idx_to_drop)

    def route_with_anchors(self):
        """
        select attractions idx for route with anchors
        """
        if self.anchors.num_anchors > 1:
            self.select_attractions_between_anchors(self.anchors)
        else:
            self.chosen_idx = [self.anchors.keys_list[0]]
        self.attractions_before_anchore(self.anchors)
        self.attractions_after_anchors(self.anchors)

    def route_without_anchors(self):
        """
        return idx list of the chosen attractions
        """
        self.first_attraction()
        # drop the already selected attraction from the list of attractions (self.chosen_idx)
        idx_to_drop = self.chosen_idx.copy()
        self.select_attractions_idx(self.tot_num_attractions - 1, idx_to_drop)

    def idx_to_dataframe(self):
        """
        receive a list of the chosen idx and retrieve the attractions in a dataframe
        """
        selected_attractions_df = self.df.iloc[self.chosen_idx].drop(columns=["index", "Unnamed: 0"])
        return selected_attractions_df

    def build_route(self):
        """
        return a dataframe of the selected route
        """
        if self.anchors:
            self.route_with_anchors()
            print("final attractions:", self.chosen_idx)
            return self.idx_to_dataframe()
        else:
            self.route_without_anchors()
            print("final attractions:", self.chosen_idx)
            return self.idx_to_dataframe()


def main():
    # read files to dataframes
    df = pd.read_csv("berlin_preprocess.csv")
    df_distances_norm = pd.read_csv("berlin_distances_norm.csv")
    df_similarity_norm = pd.read_csv("berlin_similarity_norm.csv")
    df_popularity_vec = pd.read_csv("berlin_popularity_vec.csv")
    df_tags_vec = pd.read_csv("berlin_tags_vec.csv")

    # define parameters
    num_attractions = 7
    anchors = Anchors({10: 1, 31: 7})
    chosen_tags = ["Architecture",'Culinary Experiences', "Shopping", "Art", "Urban Parks", "Museums"]

    # select formula weights
    weight_dict = {"popular": 3, "distance": 2, "similarity": 1, "tags": 1}

    new_route = RouteBuilder(df, chosen_tags, df_popularity_vec, df_similarity_norm, df_distances_norm,
                             num_attractions, weight_dict, anchors)

    new_route_df = new_route.build_route()
    print(new_route_df)


if __name__ == '__main__':
    print('hi')
    main()

