"""Tests for data_prep.py. They use small synthetic frames, so no download is needed."""
import numpy as np
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


def test_all_window_size_leakage_features_are_excluded():
    assert set(data_prep.EXCLUDED_FEATURES) == {
        "Number", "Tot sum", "ack_count", "rst_count", "syn_count", "fin_count", "Tot size"}


def test_quantised_features_are_28_and_not_excluded():
    q = set(data_prep.QUANTISED_FEATURES)
    assert len(q) == len(data_prep.QUANTISED_FEATURES) == 28
    assert not q & set(data_prep.EXCLUDED_FEATURES)
    assert not q & {"Rate", "IAT", "Std", "Variance"}


def test_quantise_rounds_grid_features_and_leaves_the_other_four():
    df = pd.DataFrame({
        "Header_Length": [28.43, 0.32], "AVG": [170.55, 63.99], "TCP": [0.97, 0.01],
        "Rate": [12.3456, 0.0123], "IAT": [0.012345, 0.000123],
        "Std": [1.2345, 0.0], "Variance": [1.52399, 0.0],
    })
    out = data_prep.quantise(df)
    for c in ("Header_Length", "AVG", "TCP"):
        assert np.allclose(out[c], df[c].round(1))
    for c in ("Rate", "IAT", "Std", "Variance"):
        assert out[c].equals(df[c])
    assert df["Header_Length"].iloc[0] == 28.43  # input not modified


def test_quantise_ignores_missing_columns():
    df = pd.DataFrame({"Rate": [1.234]})
    assert data_prep.quantise(df).equals(df)


def test_load_split_quantises(tmp_path):
    df = make_frame().assign(Header_Length=[28.43, 0.32, 1.0, 9.99, 0.05, 2.0])
    df.to_parquet(data_prep.split_path("test", tmp_path))
    out = data_prep.load_split("test", data_dir=tmp_path)
    assert np.allclose(out["Header_Length"], out["Header_Length"].round(1))
    assert out["Rate"].tolist() == [1.0, 2.0, 4.0, 5.0]


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
