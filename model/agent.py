# -*- coding: utf-8 -*-
# @File: agent.py
# @Author: Xiaocheng Tang
# @Date:   2020-03-17 17:03:34
import os,time
import _pickle as pickle
from collections import defaultdict,namedtuple

import numpy as np
import pandas as pd
from itertools import product

from grid import Grid
from scipy.optimize import linear_sum_assignment

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelfile.pkl')

class Agent(object):
  """ Agent for dispatching and reposition """

  def __init__(self):
    self.gridClass,self.workday_demand_dict, self.weekend_demand_dict = self._load(MODEL_PATH)
    self._alpha = 0.05
    self._gamma = 0.9
    self.nn_discount = 0.5
    self.grid_TW_state_value = dict()
    self.init_sv = 2.0
    self.prev_weight = 0.7
    self.demand_assigned_flag=0
    self.demand_dict = self.workday_demand_dict

  def _load(self, modelpath):
    """ Implement your model loading routine """
    with open(modelpath, 'rb') as f:
            [gridClass, workday_demand_dict,weekend_demand_dict] = pickle.load(f)
            return gridClass,workday_demand_dict,weekend_demand_dict

  def dispatch(self, dispatch_observ):
    timestamp = dispatch_observ[0]['timestamp']
    day_of_week = dispatch_observ[0]['day_of_week']
    hour,TW = self.get_Hour_TW(timestamp)
    if(self.demand_assigned_flag==0):
        self.demand_assigned_flag = 1
        if(day_of_week>=5):
            self.demand_dict = self.weekend_demand_dict
    ### Load the data
    dispatch_observ = pd.DataFrame(dispatch_observ)
    ### The Expected reward
    dispatch_observ = self.generate_expected_reward(dispatch_observ)
    ### The gridId, state_value of driver location and finish_location
    dispatch_observ = self.generate_state_reward(dispatch_observ, TW)
    ### The assignment with km algorithm
    dispatch_observ['G_reward'] = 1.0*(dispatch_observ['E_reward']+self._gamma*dispatch_observ['order_finish_gridId_reward']-dispatch_observ['driver_gridId_reward'])
    cost_matrix =  dispatch_observ.pivot_table(values='G_reward', index='order_id', columns='driver_id',fill_value=-99)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    dispatched_driver_ids = set()
    dispatch_action = []
    for index, od in zip(row_ind, col_ind):
        sub_order, sub_driver = cost_matrix.index[index], cost_matrix.columns[od]
        sub_G_reward = cost_matrix.iloc[index, od]

        if(sub_G_reward<-90): continue
        dispatched_driver_ids.add(sub_driver)
        ### The assignment
        dispatch_action.append(dict(order_id=sub_order, driver_id=sub_driver))
        ### The updating reward
        driver_gridId = dispatch_observ.loc[dispatch_observ.driver_id==sub_driver,'driver_gridId'].values[0]
        order_start_gridId = dispatch_observ.loc[dispatch_observ.order_id==sub_order,'order_start_gridId'].values[0]
        self.update_state_value(driver_gridId,TW, sub_G_reward*self.nn_discount)
        self.update_state_value(order_start_gridId,hour, sub_G_reward*(1-self.nn_discount))
    
    ### The idle driver updating
    total_driver_ids =  cost_matrix.columns.values
    for sub_driver in total_driver_ids:
        if(sub_driver in dispatched_driver_ids): continue
        driver_gridId = dispatch_observ.loc[dispatch_observ.driver_id==sub_driver,'driver_gridId'].values[0]
        v0 = self.get_state_value(driver_gridId, TW)
        increment = 4.5*(self._gamma-1)*v0
        self.update_state_value(driver_gridId, TW,increment)
    return dispatch_action

  def reposition(self, repo_observ):
    hour,TW = self.get_Hour_TW(repo_observ['timestamp'])
    repo_action = []
    for driver in repo_observ['driver_info']:
        sub_driver_id = driver['driver_id']
        sub_grid_id = driver['grid_id']
        destination = sub_grid_id
        current_gridid_sv = self.get_state_value(sub_grid_id,hour)
        nn_candidates = list(self.gridClass.nearest_neighbours[sub_grid_id].keys())
        for nn in nn_candidates:
            dist = self.gridClass.nearest_neighbours[sub_grid_id][nn]
            discount = np.power(0.95, int(dist/1000)+1)
            nn_SV = self.get_state_value(nn, TW)*discount
            if((nn_SV>current_gridid_sv) & (nn_SV>1.98)):
                destination = nn
                current_gridid_sv = nn_SV
        repo_action.append({'driver_id': sub_driver_id, 'destination': destination})
    return repo_action

  def generate_state_reward(self, dispatch_dt, TW):
    dispatch_dt['driver_gridId']=dispatch_dt.apply(lambda x: self.gridClass.lookup(x.driver_location[0], x.driver_location[1]), axis=1)
    dispatch_dt['order_start_gridId']=dispatch_dt.apply(lambda x: self.gridClass.lookup(x.order_start_location[0], x.order_start_location[1]), axis=1)
    dispatch_dt['order_finish_gridId']=dispatch_dt.apply(lambda x: self.gridClass.lookup(x.order_finish_location[0], x.order_finish_location[1]), axis=1)
    ### The begin reward
    dispatch_dt['driver_gridId_reward'] = dispatch_dt.apply(lambda x: self.get_state_value(x.driver_gridId, TW), axis=1)
    ### The end reward
    dispatch_dt['order_finish_gridId_reward'] = dispatch_dt.apply(lambda x: self.get_state_value(x.order_finish_gridId, TW), axis=1)
    return dispatch_dt

  def generate_expected_reward(self, dispatch_dt):
    dispatch_dt['answer_rate'] = dispatch_dt.apply(lambda x: self.sim_answer_rate(x.order_driver_distance), axis=1)
    dispatch_dt['E_reward'] = dispatch_dt['answer_rate']*dispatch_dt['reward_units']
    return dispatch_dt

  def get_Hour_TW(self, timestamp):
    timestamp = timestamp+8*60*60
    hour,minutes = time.gmtime(timestamp).tm_hour, time.gmtime(timestamp).tm_min
    TW = hour*3+int(minutes/20)
    return hour, TW

  def sim_answer_rate(self, distance):
    answer_rate = 0.0
    if(distance<=600): answer_rate = 1.0 - 0.0126
    elif((distance>600) & (distance<=800)): answer_rate=1.0-0.0204
    elif((distance>800) & (distance<=1000)): answer_rate=1.0-0.0269
    elif((distance>1000) & (distance<=1200)): answer_rate=1.0-0.0380
    elif((distance>1200) & (distance<=1400)): answer_rate=1.0-0.0442
    elif((distance>1400) & (distance<=1600)): answer_rate=1.0-0.0524
    elif((distance>1600) & (distance<=1800)): answer_rate=1.0-0.0585
    else:answer_rate=1-0.0671
    return answer_rate

  def get_state_value(self, grid_id, TW):
    prev_TW = TW -1
    if(grid_id in self.grid_TW_state_value):
        if(TW in self.grid_TW_state_value[grid_id]):
            return self.grid_TW_state_value[grid_id][TW]
        elif(prev_TW in self.grid_TW_state_value[grid_id]):
            prev_value = self.grid_TW_state_value[grid_id][prev_TW]
            self.grid_TW_state_value[grid_id][TW] = self.prev_weight*prev_value+(1-self.prev_weight)*self.init_sv
        else:
            self.grid_TW_state_value[grid_id][TW] = self.init_sv
    else:
        self.grid_TW_state_value[grid_id] = {}
        self.grid_TW_state_value[grid_id][TW] = self.init_sv
    return self.grid_TW_state_value[grid_id][TW]

  def update_state_value(self, grid_id, TW, increment):
    if(not grid_id in self.grid_TW_state_value):
        self.get_state_value(grid_id,TW)
    elif(not TW in self.grid_TW_state_value[grid_id]):
        self.get_state_value(grid_id,TW)
    self.grid_TW_state_value[grid_id][TW] += increment*self._alpha
