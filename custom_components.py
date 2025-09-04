import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import streamlit as st

ordinal_categories = [
    ['Average (0-6)', 'Spacious (7-10)', 'Extra Spacious (\u226511)'],
    ['Basic (0-2)', 'Premium Plus (3-5)', 'Spacious (6-9)', 'Luxury (10+)'],
    ['Low-rise (\u22642)', 'Mid-rise (3-4)', 'High-rise (5-10)', 'Skyscraper (>10)'],
    ['Basic (0-3)', 'Family Size (4-7)', 'Grand Residence (8-11)', 'Estate Level (\u226512)'],
    ['1651-1690', '1691-1730', '1731-1770', '1771-1810', '1811-1850',
     '1851-1890', '1891-1930', '1931-1970', '1971-2010', '2011+'],
    ['0', '7', '6', '5', '4', '3', '2', '1'],
    ['0', '7', '6', '5', '4', '3', '2', '1']
]

def pre_preprocessing(df_model):
    df_model['type_heater'].fillna('0', inplace=True)
    df_model['basements'] = df_model['basements'].fillna('N')
    df_model['basements'] = np.where(df_model['basements'].isin(list('ABCDEFGHIJ')), 'Y', df_model['basements'])
    df_model['basements'] = np.where(df_model['basements'].isin(['0', 0]), 'N', df_model['basements'])
    df_model['topography'].fillna('F', inplace=True)
    df_model['garage_type'].fillna('U', inplace=True)
    df_model['parcel_shape'] = df_model['parcel_shape'].replace(' ', 'U').fillna('E')
    df_model['view_type'].fillna('I', inplace=True)
    df_model['interior_condition'].fillna(0, inplace=True)
    df_model['exterior_condition'].fillna(0, inplace=True)
    df_model['fireplaces'].fillna(0, inplace=True)

    df_model['total_area'] = df_model['total_area'].fillna(df_model['total_livable_area'])
    df_model['zoning'] = df_model['zoning'].astype(str).str.strip()

    df_model['geographic_ward'] = df_model['geographic_ward'].fillna(21).astype(str)
    df_model['off_street_open'] = df_model['off_street_open'].fillna(0)
    df_model['homestead_exemption'] = df_model['homestead_exemption'].fillna(0)

    df_model['total_livable_area'] = df_model['total_livable_area'].replace(0, np.nan)
    df_model['total_livable_area'] = df_model['total_livable_area'].fillna(df_model['total_area'])
    df_model['total_area'] = df_model['total_area'].replace(0, np.nan)
    df_model['total_area'] = df_model['total_area'].fillna(df_model['total_livable_area'])

    df_model['frontage'] = df_model['frontage'].fillna(16) #16
    df_model['depth'] = df_model['depth'].fillna(77.5) #77.5

    df_model['year_built'] = pd.to_numeric(df_model['year_built'], errors='coerce')
    df_model.loc[~df_model['year_built'].between(1650, 2025), 'year_built'] = np.nan
    df_model['year_built'].fillna(1925, inplace=True)

    for col in ['number_of_rooms', 'number_of_bathrooms', 'number_of_bedrooms', 'number_stories']:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

    df_model['year_built'] = pd.cut(df_model['year_built'],
        bins=[1651, 1690, 1730, 1770, 1810, 1850, 1890, 1930, 1970, 2010, float('inf')],
        labels=ordinal_categories[4])

    df_model['number_of_rooms'] = pd.cut(df_model['number_of_rooms'], [-1, 6, 10, float('inf')], labels=ordinal_categories[0])
    df_model['number_of_bedrooms'] = pd.cut(df_model['number_of_bedrooms'], [-1, 3, 7, 11, float('inf')], labels=ordinal_categories[3])
    df_model['number_of_bathrooms'] = pd.cut(df_model['number_of_bathrooms'], [-1, 2, 5, 9, float('inf')], labels=ordinal_categories[1])
    df_model['number_stories'] = pd.cut(df_model['number_stories'], [-1, 2, 4, 10, float('inf')], labels=ordinal_categories[2])

    def binary_fireplace(x):
        return 1 if x > 0 else 0

    df_model['fireplaces'] = pd.to_numeric(df_model['fireplaces'], errors='coerce').fillna(0)
    df_model['has_fireplace'] = df_model['fireplaces'].apply(binary_fireplace).astype(str)

    df_model['interior_condition'] = df_model['interior_condition'].astype(float).astype(int).astype(str)
    df_model['exterior_condition'] = df_model['exterior_condition'].astype(float).astype(int).astype(str)

    return df_model

def normalize_ordinal_columns(df_model):
    for col in ['number_of_rooms', 'number_of_bathrooms', 'number_stories', 'number_of_bedrooms', 'year_built']:
        df_model[col] = df_model[col].astype(str).str.replace('–', '-', regex=False).str.strip()
 
    return df_model