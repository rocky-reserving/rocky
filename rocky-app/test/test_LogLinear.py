import itertools
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("../src")

from rocky.LogLinear import LogLinear
from rocky.triangle import Triangle

n_ays = 4
n_devs = 4

@pytest.fixture
def test_triangle():
    """
    build test triangle
    """
    df = pd.DataFrame({
        '12':[10, 10, 10, 10],
        '24':[20, 20, 20, np.nan],
        '36':[30, 30, np.nan, np.nan],
        '48':[40, np.nan, np.nan, np.nan]
    }, index=[2000, 2001, 2002, 2003])
    return Triangle.from_dataframe(df=df, id="t")

@pytest.fixture
def test_incremental():
    df = pd.DataFrame(
        np.array([[10, 10, 10, 10],
                  [10, 10, 10, np.nan],
                  [10, 10, np.nan, np.nan],
                  [10, np.nan, np.nan, np.nan]]),
        index=[2000, 2001, 2002, 2003],
        columns=[12, 24, 36, 48])
    return df

@pytest.fixture
def test_ata_triangle():
    df = pd.DataFrame(
        np.array([[2, 1.5, 1.3, np.nan],
                  [2, 1.5, np.nan, np.nan],
                  [2, np.nan, np.nan, np.nan],
                  [np.nan, np.nan, np.nan, np.nan]]),
        index=[2000, 2001, 2002, 2003],
        columns=[12, 24, 36, 48])
    return df

@pytest.fixture
def test_calendar_index():
    out = np.array([[1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6],
                    [4, 5, 6, 7]])
    return out

@pytest.fixture
def test_melted():
    df = pd.DataFrame({
        'accident_period':[2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003],
        'development_period':[12, 12, 12, 12,
                              24, 24, 24, 24,
                              36, 36, 36, 36,
                              48, 48, 48, 48],
        'tri':[10, 10, 10, 10,
                10, 10, 10, 0,
                10, 10, 0, 0,
                10, 0, 0, 0]
    }).astype(float)

    return df

@pytest.fixture
def test_base_dm():
    df = pd.DataFrame({
        'tri':[10, 10, 10, 10,
                10, 10, 10, np.nan,
                10, 10, np.nan, np.nan,
                10, np.nan, np.nan, np.nan],
        'is_observed':[1, 1, 1, 1,
                       1, 1, 1, 0,
                       1, 1, 0, 0,
                       1, 0, 0, 0],
        'accident_period':[2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003,
                           2000, 2001, 2002, 2003],
        'development_period':[12, 12, 12, 12,
                              24, 24, 24, 24,
                              36, 36, 36, 36,
                              48, 48, 48, 48],
        'accident_period_2001':[0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0,
                                0, 1, 0, 0],
        'accident_period_2002':[0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0,
                                0, 0, 1, 0],
        'accident_period_2003':[0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1,
                                0, 0, 0, 1],
        'development_period_024':[0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1],
        'development_period_036':[0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1],
        'development_period_048':[0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  0, 0, 0, 0,
                                  1, 1, 1, 1]
                                  
    })
    return df

def are_triangles_equal(tri_df1:pd.DataFrame, tri_df2:pd.DataFrame) -> bool:
    """
    Check if the values in two triangles are equal, ignoring NaNs
    """
    return np.allclose(tri_df1.fillna(0).values,
                       tri_df2.fillna(0).values,
                       rtol=1e-3,
                       atol=1e-3)

def are_dfs_equal(df1:pd.DataFrame, df2:pd.DataFrame) -> bool:
    """
    Check if the values in two triangles are equal, ignoring NaNs
    """
    return np.allclose(df1.values,
                       df2.values,
                       rtol=1e-3,
                       atol=1e-3)

def test_init1(test_triangle):
    t = test_triangle
    ll = LogLinear(id='loss', tri=t)
    assert ll.id == 'loss', "LOGLINEAR-001: id not set correctly"

def test_init2(test_triangle):
    t = test_triangle
    ll = LogLinear(id='loss', tri=t)
    assert are_triangles_equal(ll.tri.df, t.df), f"""LOGLINEAR-002: triangle not set correctly:
    l1.tri:
    {ll.tri}
    
    sample tri (t):
    {t}"""