import os
import numpy as np
import pandas as pd

from eforecast.common_utils.nwp_utils import get_clear_sky

class RowDataTransformer:
    def __init__(self, static_data):
        self.transformers = dict()
        self.static_data = static_data
        self.target_variable = self.static_data['target_variable']

        self.coord = self.static_data['coord']
        self.local_timezone = self.static_data['local_timezone']
        self.site_timezone = self.static_data['site_timezone']
        self.ts_resolution = self.static_data['ts_resolution']
        self.nwp_data_merge = self.static_data['nwp_data_merge']
        self.nwp_data_compress = self.static_data['compress_data']

    def transform(self, data, inverse=False):
        transformation = self.target_variable['transformer'][0] if self.target_variable['transformer'] is not None else None
        if transformation is None:
            return data
        rate_const = self.target_variable['transformer_params'][transformation]
        if transformation == 'pv_clear_sky':
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if isinstance(data, pd.DataFrame):
                dates = data.index
            else:
                raise ValueError('Data should be dataframe')
            if os.path.exists(self.static_data['path_data'] + 'clear_sky.csv'):
                ghi_old = pd.read_csv(self.static_data['path_data'] + 'clear_sky.csv', index_col=0, parse_dates=True)
                dates_new = dates.difference(ghi_old.index)
                if dates_new.shape[0] > 0:
                    ghi = get_clear_sky(dates_new, self.coord[0], self.coord[1], self.local_timezone,
                                            self.site_timezone, self.ts_resolution)
                    ghi = pd.concat([ghi_old, ghi])
                    ghi = ghi.sort_index()
                else:
                    ghi = ghi_old
            else:
                ghi = get_clear_sky(dates, self.coord[0], self.coord[1], self.local_timezone, self.site_timezone,
                                        self.ts_resolution)
            dates_diff = dates.difference(ghi.index)
            if dates_diff.shape[0] > 0:
                ghi_new = get_clear_sky(dates_diff, self.coord[0], self.coord[1], self.local_timezone,
                                        self.site_timezone, self.ts_resolution)
                ghi = pd.concat([ghi, ghi_new])
                ghi = ghi.sort_index()
            ghi = ghi[~ghi.index.duplicated()]
            ghi.to_csv(self.static_data['path_data'] + 'clear_sky.csv', index=True)
            cs_new = ghi.loc[dates]

            rate = 1 - (rate_const * (cs_new - 390) / (895 - 390))
            data_temp = []
            for col in data.columns:
                if not inverse:
                    temp = data[col].to_frame(col).multiply(rate.rename(columns={'clear_sky': col}))
                else:
                    rate_tr = 1 / rate
                    temp = data[col].to_frame(col).multiply(rate_tr.rename(columns={'clear_sky': col}))
                data_temp.append(temp)
            data_temp = pd.concat(data_temp, axis=1)
            data_temp[data_temp < 0] = 0
            data_temp = data_temp.astype(np.float32)
            data = data_temp
        else:
            raise NotImplementedError(f'{transformation} transformation is not implemented yet')
        return data

