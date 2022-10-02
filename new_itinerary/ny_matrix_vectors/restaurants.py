import json
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Any


with open(
        r'C:\Users\user\PycharmProjects\bridgify\route_builder\new_itinerary\restaurants_data\tags_dict_restaurants.json') as json_file:
    tags_dict = json.load(json_file)
    RESTAURANTS_TAGS_LIST: List = tags_dict.keys()


class Restaurants:

    #def __init__(self, df_rest, distances_matrix, rest_location_list, df_tags_weights, RESTAURANTS_TAGS_LIST):
    def __init__(self, df_rest, distances_matrix, df_tags_weights, RESTAURANTS_TAGS_LIST, uuids_to_drop):
        self.df_rest = df_rest
        self.distances_matrix = distances_matrix.fillna(1)
        #self.rest_location_list = rest_location_list
        self.df_tags_weights = df_tags_weights
        self.RESTAURANTS_TAGS_LIST = RESTAURANTS_TAGS_LIST
        self.uuids_to_drop = uuids_to_drop
        self.rest_with_separated_tags = self.df_tags()
        self.rest_with_separated_tags["uuid"] = self.df_rest.index
        self.rest_with_separated_tags.set_index("uuid", drop=True, inplace=True)


    def df_tags(self) -> DataFrame:
        """
        Creating DataFrame with separated column for each category.
        the rows are paralleled to the ones in the df. For example, if the first attraction
        has in 'categories_list' ['Breakfast', 'Lunch'], df_tags will get 1 in column 'Breakfast' and 1 in 'Lunch'
        while the rest categories will be marked as 0

        Return:
             Dataframe with separated column for each category.
        """
        tags_dict = {tag: [] for tag in self.RESTAURANTS_TAGS_LIST}
        for tag_name in tags_dict.keys():
            for tags in self.df_rest['tags']:
                if tag_name in tags:
                    tags_dict[tag_name].append(1)
                else:
                    tags_dict[tag_name].append(0)
        return pd.DataFrame(tags_dict)

    def norm_df(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def vec_scaling(self, vec):
        chosen_median = 0.5
        difference = chosen_median // vec.median()
        if difference:
            return vec * difference
        else:
            return vec

    def popularity_vec(self):
        """
        return popularity vector. The higher the number of reviews, the lower the result
        """
        pop_vec = self.df_rest["number_of_reviews"]
        max_reviews = pop_vec.sort_values(ascending=False).values[0]
        if not max_reviews:
            max_reviews = 1
        pop_vec_norm = pop_vec.apply(lambda x: x / max_reviews)
        pop_vec_norm = 1 - pop_vec_norm
        return pop_vec_norm

    def rest_vec(self, last_uuid, rest_kind_int, idx_to_drop=list()):
        # extract the distances vector from the last attraction
        try:
            rest_dist_vec = self.distances_matrix.loc[last_uuid]
        except:
            print(f"couldn't find idx {last_uuid} in rest_distance_matrix")
            rest_dist_vec = 1
        print("dist rest vec:\n",(self.vec_scaling(rest_dist_vec).sort_values()))
        # extract the popularity vector of the restaurants
        rest_pop_vec = self.popularity_vec()
        print("pop rest vec:\n",self.vec_scaling(rest_pop_vec).sort_values())

        # extract tags vector
        rest_tags_vec = self.create_tags_vec(rest_kind_int)

        # find the best product of the 2 vectors
        # Multiplication prioritizes distance (very small values) and addition prioritizes popularity ################

        # if it's lunch, distance factor is very critical factor and therefore popularity factor is given a lower weight than breakfast and dinner
        if rest_kind_int == 1:
            pop_weight = 0.05
        else:
            pop_weight = 0.2
        score = ((self.vec_scaling(rest_dist_vec) + pop_weight * self.vec_scaling(rest_pop_vec)) * rest_tags_vec)
        idx_to_drop = list(set(idx_to_drop + self.uuids_to_drop))
        score.drop(index=idx_to_drop, inplace=True)
        #score.drop(index=self.uuids_to_drop)
        score = score[score != 0]
        return score

    def best_rest_uuid(self, rest_vec):
        print("scores:\n", rest_vec.sort_values())
        if rest_vec.sort_values().index[0] in self.uuids_to_drop:
            print("chosen!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return rest_vec.sort_values().index[0]

    def rest_between_attractions(self, idx1, idx2, idx_to_drop, rest_kind_int):  # rest_kind_int: 'breakfast': 0, 'lunch': 1, 'dinner': 2
        vec1 = self.rest_vec(idx1, rest_kind_int, idx_to_drop)
        vec2 = self.rest_vec(idx2, rest_kind_int, idx_to_drop)
        return vec1 + vec2

    def selected_rest(self, selected_uuid):
        # return self.df_rest.iloc[selected_idx].T
        return self.df_rest[self.df_rest["uuid"] == selected_uuid]

    def create_tags_vec(self, time_index):

        def create_vector(col_name):
            df = self.rest_with_separated_tags.copy()
            score_str = col_name + "_score"
            df[score_str] = 0
            for uuid in df.index:
                df[score_str].loc[uuid] = np.dot(
                    np.array(df[cols].loc[uuid]),
                    np.array(self.df_tags_weights[col_name]))
            max_score = df[score_str].max()
            df[score_str] /= max_score
            df[score_str] = 1 - df[score_str]
            second_min_score = sorted(df[score_str].unique())[1]
            df[score_str][df[score_str] == 0] = second_min_score // 2
            df[score_str] = df[score_str].apply(lambda x: 0 if x == 1 else x)
            return df[score_str]

        cols = self.rest_with_separated_tags.columns
        if time_index == 0:
            return create_vector("breakfast")

        elif time_index == 1:
            return create_vector("lunch")

        elif time_index == 2:
            return create_vector("dinner")


def df_tags(df: DataFrame, tag_col: str, tags_lst: List[str]) -> DataFrame:
    """
    Creating DataFrame with separated column for each category.
    the rows are paralleled to the ones in the df. For example, if the first attraction
    has in 'categories_list' ['Art', 'Museums'], df_tags will get 1 in column 'Art' and 1 in 'Museums'
    while the rest categories will be marked as 0

    Args:
        df: Dataframe of the attractions
        tag_col: str, the name of the column with the categories/tags
        tags_lst: list of all tags

    Return:
         Dataframe with separated column for each category.
    """
    tags_dict = {tag: [] for tag in tags_lst}
    for tag_name in tags_dict.keys():
        for tags in df[tag_col]:
            if tag_name in tags:
                tags_dict[tag_name].append(1)
            else:
                tags_dict[tag_name].append(0)
    return pd.DataFrame(tags_dict)


