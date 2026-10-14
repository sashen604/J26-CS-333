"""Tests for data_prep.py. They use small synthetic frames, so no download is needed."""
import pandas as pd
import pytest

import data_prep


def make_frame():
    return pd.DataFrame({
        "Number": [10, 100, 100, 10, 100, 100],
        "Rate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "Label": ["BenignTraffic", "Mirai-udpplain", "DDoS-SYN_Flood",
                  "BenignTraffic", "Mirai-greeth_flood", "Recon-PortScan"],
        "attack_class": ["Benign", "Mirai", "DDoS", "Benign", "Mirai", "Recon"],
        "label": [0, 1, 1, 0, 1, 1],
    })


def test_filter_keeps_only_benign_and_mirai():
    out = data_prep.filter_mirai_vs_benign(make_frame())
    assert set(out["attack_class"]) == {"Benign", "Mirai"}
    assert len(out) == 4


def test_target_is_one_for_mirai_and_zero_for_benign():
    out = data_prep.filter_mirai_vs_benign(make_frame())
    assert (out.loc[out["attack_class"] == "Mirai", data_prep.TARGET] == 1).all()
    assert (out.loc[out["attack_class"] == "Benign", data_prep.TARGET] == 0).all()


def test_filter_does_not_modify_input():
    df = make_frame()
    data_prep.filter_mirai_vs_benign(df)
    assert data_prep.TARGET not in df.columns
    assert len(df) == 6


def test_feature_columns_exclude_label_columns_and_number():
    out = data_prep.filter_mirai_vs_benign(make_frame())
    assert data_prep.feature_columns(out) == ["Rate"]


def test_number_is_an_excluded_feature():
    assert "Number" in data_prep.EXCLUDED_FEATURES


def test_class_counts():
    out = data_prep.filter_mirai_vs_benign(make_frame())
    assert data_prep.class_counts(out) == (2, 2)


def test_load_split_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_prep.load_split("train", data_dir=tmp_path)


def test_load_split_reads_and_filters_parquet(tmp_path):
    make_frame().to_parquet(data_prep.split_path("test", tmp_path))
    out = data_prep.load_split("test", data_dir=tmp_path)
    assert len(out) == 4
    assert data_prep.TARGET in out.columns


def test_split_url_points_at_random_3way():
    assert data_prep.split_url("validation").endswith("random_3way/validation-00000-of-00001.parquet")
