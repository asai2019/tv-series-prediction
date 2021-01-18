# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:53:54 2020

@author: The Prince
"""

#%% Bokeh Bar Plot
from os.path import dirname, join
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Slider
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Turbo256

# Load dataset and subset desired columns
vis = pd.read_csv('base_data.csv', index_col='tconst').dropna(subset=['genres']).loc[:,['startYear','genres']]
# Split up genres column
vis[['genre1','genre2','genre3']] = vis['genres'].str.split(',').apply(pd.Series).rename(columns={0:'genre1',1:'genre2',2:'genre3'})
# Melt table and then pivot to get desired format
ds = pd.melt(vis, id_vars='startYear', value_vars=['genre1','genre2','genre3'], value_name='genre').groupby(by=['startYear','genre'],as_index=False).count()

unique_genres = ['Action','Adventure','Animation','Biography',
                 'Comedy','Crime','Documentary','Drama','Family',
                 'Fantasy','Film-Noir','Game-Show','History',
                 'Horror','Music','Musical','Mystery','News',
                 'Reality-TV','Romance','Sci-Fi','Short','Sport',
                 'Talk-Show','Thriller','War','Western']

index_cmap = factor_cmap('genre', palette=Turbo256[0::9], 
                          factors=unique_genres)

desc = Div(text=open(join(dirname(__file__), "description2.html")).read(), sizing_mode="stretch_width")

# Create Input controls
startYearControl = Slider(title="Start Year", start=1906, end=2026, value=1906, step=1)   

# Create Column Data Source to be used by plot
source = ColumnDataSource(data=ds)

TOOLTIPS = [
        ('Genre','@genre'),
        ('Count','@variable')
        ]

p = figure(x_range=unique_genres, plot_height=800, plot_width=1300, title="", sizing_mode='scale_both', tooltips=TOOLTIPS)
p.vbar(x='genre', top='variable', source=source, width=0.75, fill_color=index_cmap, line_color=None, alpha=0.8)
p.axis.major_label_text_font_size = "12pt"
p.yaxis.axis_label = 'Count'
p.yaxis.axis_label_text_font_size = "12pt"
p.y_range.start = 0
p.xaxis.axis_label = 'Genre'
p.xaxis.axis_label_text_font_size = "12pt"
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = np.pi / 6
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None


def select_series():
    selected = ds[ds.startYear == startYearControl.value]
    return selected


def update():
    df = select_series()
    p.title.text = 'TV Series Genre Distribution by Start Year: {} TV series debuted in {}'.format(df.variable.sum(), startYearControl.value)
    p.title.text_font_size = '20pt'
    p.x_range.factors = unique_genres
    source.data = df

    
controls = [startYearControl]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(desc,*controls, width=500, height=700)
inputs.sizing_mode = 'fixed'
l = layout([[inputs,p]],
           sizing_mode='scale_both')

update()

curdoc().add_root(l)
curdoc().title = 'TV Series Genres by Year Visualization'  