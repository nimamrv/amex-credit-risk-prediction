"""Generate fictional customer statements for a small execution demo."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n_customers=120, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range('2017-04-01', periods=12, freq='MS')
    for i in range(n_customers):
        target = int(i % 2 == 0)
        for date in dates:
            rows.append({'customer_ID': f'fictional_{i:05d}', 'S_2': date.strftime('%Y-%m-%d'),
                         'target': target, 'P_2': float(rng.normal(.6-.12*target,.12)),
                         'B_1': float(rng.normal(.15+.09*target,.05)),
                         'D_39': float(rng.normal(.1+.04*target,.03)),
                         'B_30': int(rng.integers(0,3))})
    return pd.DataFrame(rows)

if __name__ == '__main__':
    out = Path('data/synthetic_customers.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} fictional monthly records to {out}')
