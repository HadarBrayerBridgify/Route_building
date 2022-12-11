import pytest
import json
import multiple_days_route_anchors as new_ip


# setup
with open("config.json") as config_file:
    config = json.load(config_file)
route_object = new_ip.main(config)
count_reviews = route_object.df["number_of_reviews"].sort_values(ascending=False)


def test_popularity_vec_len():
    assert len(route_object.popularity_vec) == len(route_object.df)


def test_popularity_vec_range():
    assert route_object.popularity_vec.max() == 1
    assert route_object.popularity_vec.min() == 0


def test_popularity_vec_nan():
    assert route_object.popularity_vec.isna().sum() == 0


def test_popularity_vec_one_one_value():
    assert len(route_object.popularity_vec[route_object.popularity_vec == 0]) == 1


def test_popularity_vec_not_binary():
    assert len(route_object.popularity_vec.value_counts()) > 2


def test_popularity_vec_nan_number_reviews():
    nan_uuids = list(route_object.df["uuid"][route_object.df["number_of_reviews"].isna()].values)
    nan_uuids += list(route_object.df["uuid"][route_object.df["number_of_reviews"] == 0].values)
    popularity_one = route_object.popularity_vec[route_object.popularity_vec == 1].index
    assert len(set(nan_uuids) - set(popularity_one)) == 0


def test_popularity_vec_max_number_reviews():
    sort_reviews = route_object.df["number_of_reviews"].sort_values(ascending=False)
    if sort_reviews.values[0] > 50:
        max_uuid = route_object.df.loc[sort_reviews.index[0]]["uuid"]
        best_popularity_uuids = route_object.popularity_vec.sort_values()[:10].index
        assert max_uuid in best_popularity_uuids