import os
import pandas as pd
import re
import requests
from sklearn.preprocessing import StandardScaler

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"

def clean_weight(weight_string: str) -> float:
    """Finds weight in kilograms using regex and returns it as float"""
    return float(re.search('[.0-9]+ kg.', weight_string).group()[:-4])


def clean_data(path: str) -> pd.DataFrame:
    # 1 Read the csv.
    df = pd.read_csv(path)
    # 2 Birthday and draft year to datetime objects
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    # 3 Filling NaNs in 'team' with 'No team'
    df['team'].fillna('No Team', inplace=True)
    # 4-7
    df['height'] = df['height'].agg(lambda x: float(x[-4:]))
    df['weight'] = df['weight'].agg(clean_weight)
    df['salary'] = df['salary'].agg(lambda x: float(x.strip('$')))
    # 8-9
    df['draft_round'] = df['draft_round'].agg(lambda x: '0' if x == 'Undrafted' else x)
    df['country'] = df['country'].agg(lambda x: x if x == 'USA' else 'Not-USA')
    return df


def feature_data(cdf: pd.DataFrame) -> pd.DataFrame:
    # 2 Parse version as datetime
    cdf['version'] = pd.to_datetime(cdf['version'].apply(lambda x: x[-2:]), format='%y')
    # 3.1 Calculate age as timedelta
    cdf['age'] = pd.DatetimeIndex(cdf['version']).year - pd.DatetimeIndex(cdf['b_day']).year
    # 4 Same as above for experience
    cdf['experience'] = pd.DatetimeIndex(cdf['version']).year - pd.DatetimeIndex(cdf['draft_year']).year
    # 5
    cdf['bmi'] = cdf['weight'] / cdf['height'] ** 2
    # 6
    cdf.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1, inplace=True)
    # 7
    indices_to_drop = cdf.columns[cdf.nunique() > 50].intersection(cdf.select_dtypes(include='object').columns)
    cdf.drop(columns=indices_to_drop, inplace=True)
    return cdf


def multicol_data(features: pd.DataFrame) -> pd.DataFrame:
    """Checks for multicollinearity and drops corresponding features"""
    # Calculate correlation matrix for numerical features excluding target (salary)
    # and transform to df of bools where True is for absolute value of correlation
    # coefficient is larger than 0.5
    corr_df = abs(features[['rating', 'age', 'experience', 'bmi']].corr()) > 0.5

    # Iterate through correlation matrix and add the features in question to a set
    high_corr_feats = set()
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if i < j and corr_df.iloc[i][j]:
                high_corr_feats.add(corr_df.columns[i])
                high_corr_feats.add(corr_df.index[j])

    # For each feature in set, calculate correlation with target
    corr_with_target = {}
    for feature in high_corr_feats:
        corr_with_target.update({feature: features[['salary', feature]].corr().iloc[0, 1]})
    # Finally, drop the feature with lower coefficient and return the resulting df
    return features.drop(columns=min(corr_with_target))

def transform_data(X: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    y = X['salary']
    X = X.drop('salary', axis=1)
    # Split the data into numeric and categorical columns
    X_num = X.select_dtypes(include='number')
    X_cat = X.select_dtypes(exclude='number')

    # use get_dummies on categories
    X_cat_encoded = pd.get_dummies(X_cat)

    # Fit and transform the numeric columns using a StandardScaler
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine the scaled numeric columns and encoded categorical columns
    X_final = pd.concat([pd.DataFrame(X_num_scaled, columns=X_num.columns), X_cat_encoded], axis=1)
    # clean column names by hand
    X_final.columns = X_final.columns.map(lambda x: x.removeprefix('team_'))
    X_final.columns = X_final.columns.map(lambda x: x.removeprefix('position_'))
    X_final.columns = X_final.columns.map(lambda x: x.removeprefix('country_'))
    X_final.columns = X_final.columns.map(lambda x: x.removeprefix('draft_round_'))

    # Return the transformed data and the target variable
    return X_final, y

df = multicol_data(feature_data(clean_data(data_path)))
X, y = transform_data(df)
answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }
print(answer)