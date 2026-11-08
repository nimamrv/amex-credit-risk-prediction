"""Customer-level features derived strictly from observations at/before cutoff."""
import numpy as np
import pandas as pd

CATEGORICAL = ['B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68']

def build_features(data: pd.DataFrame, cutoff='2018-03-31') -> pd.DataFrame:
    required = {'customer_ID','S_2','target'}
    if not required.issubset(data):
        raise ValueError(f'Missing columns: {required-set(data.columns)}')
    df = data.copy()
    df['S_2'] = pd.to_datetime(df['S_2'], errors='raise')
    cutoff = pd.Timestamp(cutoff)
    df = df.loc[df['S_2'].le(cutoff)].sort_values(['customer_ID','S_2']).copy()
    if df.empty: raise ValueError('No rows at or before cutoff')
    if df.groupby('customer_ID')['target'].nunique(dropna=False).gt(1).any():
        raise ValueError('Inconsistent target labels within customer')
    numeric = [c for c in df.select_dtypes(include='number').columns if c not in CATEGORICAL+['target']]
    groups = df.groupby('customer_ID', sort=True)
    base = groups[numeric].agg(['mean','sum','min','max','std'])
    base.columns = [f'{c}_{stat}' for c,stat in base.columns]
    pieces = [base]
    for months in (3,6,9,12):
        recent = df.loc[df['S_2'].gt(cutoff-pd.DateOffset(months=months))]
        g = recent.groupby('customer_ID')
        roll = g[numeric].agg(['mean','min','max','std'])
        roll.columns = [f'{c}_{stat}_{months}m' for c,stat in roll.columns]
        pieces.append(roll)
        cats = [c for c in CATEGORICAL if c in recent.columns]
        if cats:
            encoded = pd.get_dummies(recent[cats].astype('string'), prefix=cats, dtype=float)
            encoded['customer_ID'] = recent['customer_ID'].values
            egroup = encoded.groupby('customer_ID')
            rate = egroup.mean().add_suffix(f'_response_rate_{months}m')
            ever = egroup.max().add_suffix(f'_ever_response_{months}m')
            pieces.extend([rate,ever])
    features = pd.concat(pieces,axis=1)
    last = groups['S_2'].max()
    features['days_since_last_statement'] = (cutoff-last).dt.days
    # The original D/R ratio mixed scaled heterogeneous fields; omitted pending domain validation.
    labels = groups['target'].first().rename('target')
    result = features.join(labels).reset_index()
    result = result.replace([np.inf,-np.inf],np.nan)
    return result
