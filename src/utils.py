import ciw
from ciw.dists import *
import math
import json
from collections import defaultdict
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown, HTML

import datetime
import zipfile

import pickle
import pandas as pd
import numpy as np
import random

from typing import List, Dict
from ciw.simulation import Simulation

import plotly

TICKS_IN_HOUR = 60
TICKS_IN_DAY = 24 * TICKS_IN_HOUR
TICKS_IN_YEAR = 365 * TICKS_IN_DAY
CUR_COLOR = '#2E7D32'
PREV_COLOR = '#A5D6A7'
INDEX_TRIAGE_NODE = 1
INDEX_INV_NODE = 2


def calc_throughput(_q: Simulation) -> float:
    '''
    Calculate and return the throughput of a given Simulation
    '''
    return len(_q.get_all_records()) / _q.current_time
    
def calc_utilization(_q: Simulation, util_index: int=INDEX_TRIAGE_NODE) -> int:
    '''
    Calculate and return the utilization of a node given a Simulation object.
    
    NOTE: Will detect if it was supplied an Exit Node (which has no utilization),
    and just return None. Makes the code calling this function less complex.
    '''
    if str(_q.nodes[util_index]) == 'Exit Node': 
        return None
    return _q.nodes[util_index].server_utilisation * 100

def make_indicator(current: int, previous: int=None, gauge: bool=False, suffix='/min') -> plotly.graph_objs.Indicator:
    '''
    Makes a plotly indicator figure. Callers can draw gauges, and specify suffixes.
    '''
    kwargs = {"mode": "number+delta",
              "value": current,
              "number": {'suffix': suffix},
              "domain": {'x': [0, 1], 'y': [0, 1]}}
              
    if previous is not None:
        kwargs["delta"] = {'position':'bottom','reference': previous}
              
    if gauge:
        kwargs['mode'] = "gauge+number+delta" if previous else "gauge+number"
        kwargs['gauge'] = {'axis': {'range': [0, 100]},
                           'steps' : [ {'range': [0, 70], 'color': "lightgray"},
                                       {'range': [70, 90], 'color': "gray"}],
                            'threshold' : { 'line': {'color': "red", 'width': 4}, 
                                             'thickness': 0.75, 'value': 90}}
    return go.Indicator(**kwargs)

def build_df(q:Simulation, test:pd.DataFrame) -> pd.DataFrame:
    '''
    Build a DataFrame of the various values we need. The DataFrame is used by all other code
    to build various aspects of the dashboard.
    '''
    df = pd.DataFrame([{
        'date': test['ds'].iloc[0] + datetime.timedelta(minutes=rec.arrival_date),
        'exit_date': rec.exit_date,
        'arrival_date': rec.arrival_date,
        'latency': rec.exit_date -  rec.arrival_date,
        'customer_class': rec.customer_class,
        'wait_time': rec.waiting_time 
    } for rec in  q.get_all_records()])
    df['date'] = pd.to_datetime(df['date']).dt.round('s')
    df.sort_values(by='date', inplace=True)
    return df

def make_histogram(values: List, name: str, marker_color: str, nbins:int= 20, bingroup=None, **kwargs) -> plotly.graph_objs.Histogram:
    '''
    Make a histogram figure. Will return max count which can be used to draw slo lines.
    '''
    counts, _ = np.histogram(values, bins=nbins)
    return (go.Histogram(x=values, 
                       name=name,
                       marker_color=marker_color,
                       bingroup=bingroup, **kwargs), max(counts))


def make_ts(x, y, line=dict(color=CUR_COLOR,width=1)) -> plotly.graph_objs.Scatter:
    '''
    Make a time series figure.
    '''
    return go.Scatter(x=x, y=y,
                      showlegend=False,
                      line=line)

class Dashboard(object):
    '''
    Class is used to abstract away some of the intricacies of building plotly subplots.
    '''
    def __init__(self, title):
        self.title = title
        self.titles = []
        self.specs = []
        self.figures = []
        self.num_cols = 0
        
    def add_row(self, titles, cell_type, figures, colspan=None):
        '''
        Add a row to the dashboard. This method under the hood, will
        recalculate max number of columns, handle column spanning, and updating
        figure specs.
        '''
        if type(titles) != list:
            titles = [titles]    
        self.titles.extend(titles)
        
        if type(figures) != list:
            figures = [figures]
        
        if self.num_cols < len(titles):
            self.num_cols = len(titles)
        
        if cell_type == 'domain':
            self.specs.append([{"type": cell_type} for i in range(len(titles))])
        elif cell_type == 'xy':
            if colspan is None:
                colspan = self.num_cols 
            tup = (None, ) * (colspan-1)
            
            self.specs.append([{"type": "xy", "colspan": colspan}, *tup])
        else:
            raise Exception(f"Currently dont support cell_type={cell_type}")
        
        for idx, fig in enumerate(figures):
            self.figures.append((fig, len(self.specs), idx+1))
            
    def insert_figure(self, row, col, figure):
        '''
        Add a figure to an already "added row" this is if you are adding
        a trace to overlay lines/bars etc you'll call insert after you've
        called add_row(..)
        '''
        self.figures.append((figure, row, col))
        
    def draw_slo(self, slo, top_y, row, col, showlegend=True):
        '''
        Draw a read line to represent the service level obejctive and
        cap the line using top_y val.
        '''
        self.figures.append((go.Scatter(x=[slo,slo], y=[0,top_y],
                                mode="lines+text",
                                name="SLO",
                                showlegend=showlegend,
                                text=[None, "SLO"],
                                textposition="bottom center",
                                line=dict(color='red',width=3)), row, col))
        
    def build(self):
        '''
        This method is used once you've filled out your plotly subplot. It will
        build the subplot with all the figures and return the main figure object.
        '''
        main_fig = make_subplots(rows=len(self.specs), cols=self.num_cols,
                         subplot_titles=self.titles,
                         specs=self.specs)

        for (fig, row, col) in self.figures:
            main_fig.add_trace(fig, row=row, col=col)   
            
        main_fig.update_layout(barmode="overlay", bargap=0.1, 
                       height=900, width=800,
                       title={ 'text': self.title,
                       'y':0.98, 'x':0.5, 'xanchor': 'center',
                       'font': {'size': 20}, 'yanchor': 'top'},yaxis3=dict(rangemode='tozero'))

        
        
        main_fig['layout']['annotations'][0].update({'y':1.03})
        main_fig['layout']['annotations'][1].update({'y':1.03})
        if self.num_cols == 3:
            main_fig['layout']['annotations'][2].update({'y':1.03})
        return main_fig
        

def build_dashboard(q, prev_q=None, test=None, wait_time_slo=15, latency_slo=30, title="Run Data", show_slo=False):
    '''
    This method builds the dashboards used throughout the presentation.
    
    Args:
        q (Simulation): The current simulation that was run.
        prev_q (Simulation, optional): The previous simulation run, if specified will contrast current run with previous.
        wait_time_slo (int): The wait time SLO 
        latency_slo (int): The latency SLO
        title (str): Title of the dashboard
        show_slo(bool): If true will draw SLO line on latency and wait time histograms.
        
    Returns:
        A plotly figure that is our simulation dashboard. Callers are responsible for calling "show"
    '''
    prev_throughput = prev_utilization = utilization_inv = ut_inv_indicator = None
    top_y_lat = top_y_lat_prev = top_y_wait = top_y_wait_prev = 0
    
    # Create the dashboard object
    db = Dashboard(title)
    
    # Calculate throughput and utliziation
    throughput = calc_throughput(q)
    utilization = calc_utilization(q)
    
    # Calculate if we have a previous Q
    if prev_q != None:
        prev_throughput = calc_throughput(prev_q)
        prev_utilization = calc_utilization(prev_q)
        
    # Build our indicator images
    tp_indicator = make_indicator(throughput, previous=prev_throughput)
    ut_indicator = make_indicator(utilization, previous=prev_utilization, gauge=True, suffix='%')
    
    # Initialize our figures, and titles this currently assumes we aren't using a simulation with "tiering"
    figures = [tp_indicator, ut_indicator]
    titles = ["Throughput","Utilization"]
    
    # If we have 5 nodes we are in a simulation with tiering and want to break out by Triage and Investigation
    if len(q.nodes) == 5:
        # Update titles, figures, and build utilization indicator figure
        titles = ["Throughput","Utilization (Triage)", "Utilization (Investigation)"]
        utilization_inv = calc_utilization(q, util_index=INDEX_INV_NODE)
        ut_inv_indicator = make_indicator(utilization_inv, previous=calc_utilization(prev_q, util_index=INDEX_INV_NODE), gauge=True, suffix='%')
        figures.append(ut_inv_indicator)

    # Add the rows
    db.add_row(titles, "domain", figures)
    
    # Build dataframe of current simulation
    df_q = build_df(q, test)

    # Make latency histogram
    h1, top_y_lat = make_histogram(df_q[(df_q['exit_date'] > df_q['arrival_date'])]['latency'].values, "Current", CUR_COLOR, nbins=20, bingroup=1, nbinsx=20)
    db.add_row("Latency Distribution", "xy", h1)
    
    # Make wait time histogram
    h2, top_y_wait = make_histogram(df_q[(df_q['wait_time'] > 0)]['wait_time'], "Current", CUR_COLOR, nbins=20, bingroup=2, showlegend=False)
    db.add_row("Wait Time Distribution", "xy", h2)
            
    # Build arrival rate scatter plot
    arrival = pd.Series(np.ones(len(df_q)), index=df_q['date']).resample('5T').sum()
    ts1 = make_ts(arrival.index, arrival)
    ts_actual = go.Scatter(
            x=test['ds'], y=test['y'], name='Actual values',
            line=dict(color='red',width=1)
        )

    db.add_row("Arrival Rate", "xy", ts1)
    #db.add_row("Arrival Rate", "xy", ts_actual)
    
    # Build latency scatter plot
    ts1 = make_ts(df_q['date'], df_q['latency'])
    db.add_row("Latency", "xy", ts1)
    
    if prev_q is not None:
        # If we have a previous q object, then do the same with that using the PREV_COLOR to differentiate
        df_pq = build_df(prev_q, test)
        ph1, top_y_lat_prev = make_histogram(df_pq[(df_pq['exit_date'] > df_pq['arrival_date'])]['latency'].values, "Prev", PREV_COLOR, nbins=20, bingroup=1, nbinsx=20)
        db.insert_figure(row=2, col=1, figure=ph1)
        
        # Add wait time
        ph2, top_y_wait_prev = make_histogram(df_pq[(df_pq['wait_time'] > 0)]['wait_time'], "Prev", PREV_COLOR, nbins=20, bingroup=2, showlegend=False)
        db.insert_figure(row=3, col=1, figure=ph2)
        
        # Add arrival times overlay
        arrival = pd.Series(np.ones(len(df_pq)), index=df_pq['date']).resample('5T').sum()
        ts2 = make_ts(arrival.index, arrival, line=dict(color=PREV_COLOR,width=1))
        db.insert_figure(row=4, col=1, figure=ts2)
        
        # Add latency overlay
        ts2 = make_ts(df_pq['date'], df_pq['latency'], line=dict(color=PREV_COLOR,width=1))
        db.insert_figure(row=5, col=1, figure=ts2)
    
    if show_slo:
        db.draw_slo(latency_slo, max([top_y_lat, top_y_lat_prev]), row=2, col=1)
        db.draw_slo(wait_time_slo, max([top_y_wait, top_y_wait_prev]), row=3, col=1, showlegend=False)

    
    return db.build()


def get_arr_counts(df):
    df = df.copy()
    df['y'] = [12 / y if y!= 0 else 0 for y in df['y'] ] # minutes by job
    df['y'] = df['y'].fillna(0.)
    dd = df.groupby([df.ds.dt.dayofweek, df.ds.dt.hour]).mean(numeric_only=True)
    dd = dd.rename_axis(['day', 'hour']).to_dict()['y']
    # Build look up that is keyd on day of week -> hour -> counts
    counts = {day: {hour : 0.0 for hour in range(24)} for day in range(7)} 
    # Iterate over pickle file, we have day as an index, and then sub index of hours : counts
    for (day, hr), cnts in dd.items():
        counts[day][hr] = 1 / cnts
    return counts


class HistoricalDistribution(ciw.dists.Distribution):
    '''
    Object is used by CIW to simulate our arrival rates
    '''
    def __init__(self, data, adjustment_factor=0.0):
        self.dists = defaultdict(dict)
        
        # Get counts from historical distribution
        counts = get_arr_counts(data)
        # Build our Expeonential distributions off the arrival rate
        for day in counts.keys():
            for hr, avg_per_min in counts[day].items():
                self.dists[day][hr] = ciw.dists.Exponential(avg_per_min -(avg_per_min * adjustment_factor) )
        
    def sample(self, t, ind=None):
        # Method called by CIW
        day    = math.floor((t / TICKS_IN_DAY) % 7)
        hour   = math.floor((t / TICKS_IN_HOUR) % 24)
        sample = self.dists[day][hour].sample(t,ind)
        return sample

class ServiceDistribution(ciw.dists.Distribution):
    '''
    Object used by queue to simulate service times 
    '''
    def __init__(self, avg_time_to_triage, avg_time_to_investigate,
                 avg_time_to_report, prob_of_inv=.2):
        self.time_to_triage      = Exponential(1/avg_time_to_triage)
        self.time_to_investigate = Exponential(1/avg_time_to_investigate)
        self.time_to_report      = Exponential(1/avg_time_to_report)
        self.prob_of_inv         = prob_of_inv

    def sample(self, t, ind=None):
        total_time = self.time_to_triage.sample()
        if random.random() < self.prob_of_inv:
            total_time += self.time_to_investigate.sample()
            total_time += self.time_to_report.sample()

        return total_time   
