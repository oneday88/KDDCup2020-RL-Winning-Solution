from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

lon_factor = 0.73   # math.cos(mean_lat=30.673)

class Grid:
    def __init__(self, grid_table, idle_prob_table, dist_threshold=2100):
        ### Get the initial grids
        self.grids = dict()
        self.nearest_neighbours = dict()
        ### The preprocess of grid_table
        grid_table.columns = ['gridId', 'lonVertex1','latVertex1','lonVertex2','latVertex2','lonVertex3','latVertex3','lonVertex4','latVertex4','lonVertex5','latVertex5','lonVertex6','latVertex6']
        grid_table['lonCenter'] = grid_table[['lonVertex1','lonVertex2','lonVertex3','lonVertex4','lonVertex5','lonVertex6']].mean(axis=1).values
        grid_table['latCenter'] = grid_table[['latVertex1','latVertex2','latVertex3','latVertex4','latVertex5','latVertex6']].mean(axis=1).values
        grid_table = grid_table[['gridId','lonCenter','latCenter']]

        for index, row in grid_table.iterrows():
            ### The loopup table
            grid_id = row['gridId']
            lonCenter,latCenter=row['lonCenter'],row['latCenter']
            self.grids[grid_id] = (lonCenter*lon_factor,latCenter)
            ### The nearest neighbours
            dist = self.vectorize_lonlat_distance(lonCenter,latCenter,grid_table['lonCenter'].values, grid_table['latCenter'].values)
            pairwise_gridId_list = grid_table['gridId'].values
            pairwise_gridId_list= pairwise_gridId_list[dist<dist_threshold]
            dist = dist[dist<dist_threshold]
            assert len(pairwise_gridId_list)>0
            self.nearest_neighbours[grid_id] = {}
            for i in range(len(pairwise_gridId_list)):
                self.nearest_neighbours[grid_id][pairwise_gridId_list[i]] = dist[i]

        assert len(self.grids) == 8518
        self.grid_ids = list(self.grids.keys())  # type: List[str]
        self.kdtree = KDTree(list(self.grids.values()))
        ### The nearest neighbours
        dist = self.vectorize_lonlat_distance(lonCenter,latCenter,grid_table['lonCenter'].values, grid_table['latCenter'].values)
        pairwise_gridId_list = grid_table['gridId'].values
        pairwise_gridId_list= pairwise_gridId_list[dist<dist_threshold]

        ### The preprocess of idel_prob table
        self.idle_prob_table = self.idle_init_preprocess(idle_prob_table, grid_table)
    
    def idle_init_preprocess(self, idle_prob_table, grid_table):
        idle_prob_table.columns = ['hour','start_grid_id','end_grid_id','trans_prob']
        grid_table.columns = ['start_grid_id','start_lon','start_lat']
        idle_prob_table =pd.merge(idle_prob_table, grid_table, on=['start_grid_id'])
        grid_table.columns = ['end_grid_id','end_lon','end_lat']
        idle_prob_table =pd.merge(idle_prob_table, grid_table, on=['end_grid_id'])
        dist = self.vectorize_lonlat_distance(idle_prob_table.start_lon.values, idle_prob_table.start_lat.values, idle_prob_table.end_lon.values,
                idle_prob_table.start_lat.values)
        idle_prob_table['transit_distance'] = dist
        idle_prob_table['transit_time'] = idle_prob_table['transit_distance']/3
        return idle_prob_table[['hour','start_grid_id','end_grid_id','end_lon','end_lat','trans_prob','transit_distance','transit_time']]

    def lookup(self, lng: float, lat: float) -> str:
        _, i = self.kdtree.query([lng*lon_factor, lat])
        return self.grid_ids[i]

    def vectorize_lonlat_distance(self, lon1, lat1, lon2, lat2):
        radius = 6371*1000 # m
        dlat = np.radians(lat2-lat1)
        dlon = np.radians(lon2-lon1)

        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = radius * c
        return d.astype(int)
