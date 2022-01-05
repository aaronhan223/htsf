import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import s3fs


def preprocessing(s3_input):

    s3 = s3fs.S3FileSystem()
    raw_data = pq.ParquetDataset(s3_input, filesystem=s3).read_pandas().to_pandas()
    pd.set_option('display.max_columns', None)
    pd.options.display.float_format = '{:.2f}'.format

    raw_data['amount'] = raw_data['amount'].astype(np.float)
    raw_data['tx_date'] = pd.to_datetime(raw_data['tx_date'])
    raw_data['end_date'] = pd.to_datetime(raw_data['end_date'])
    raw_data['rank'] = raw_data['rank'].apply(lambda x: str(int(x)))
    raw_data = raw_data[['realm_id', 'tx_date', 'end_date', 'amount', 'rank']]

    data = raw_data.copy()
    end_data = data[['realm_id', 'end_date', 'rank']].drop_duplicates()
    end_data = end_data.rename(columns={"end_date": "tx_date"})
    end_data.loc[:, 'amount'] = 0.0
    end_data.loc[:, 'end_date'] = end_data.loc[:, 'tx_date']

    data = data.append(end_data)
    data = data.groupby(['realm_id', 'tx_date', 'end_date', 'rank']).agg({'amount': np.sum}).reset_index()
    data = ( 
        data
        .groupby(['realm_id', 'rank'])
        .apply(lambda x : x.resample("W-FRI", on='tx_date').amount.sum())
        .reset_index()
        .pivot_table(index=['realm_id', 'tx_date'], columns='rank', values='amount')
        .sort_index()
        .fillna(0.0)
    )
    data.loc[:, '0'] = data.apply(lambda x: np.sum(x), axis=1)
    data = data.reset_index()

    return data, [[4]]
