# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:18:43 2020

@author: The Prince
"""

#%% Bokeh Scatter Plot
from os.path import dirname, join
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, LabelSet, Select, Slider, TextInput
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Set1

# Prior settings (for genres and embedding clusters)
vis = pd.read_csv('bokeh_vis.csv')
vis.loc[:,'c0_prob':'c6_prob'] = vis.loc[:,'c0_prob':'c6_prob'].apply(lambda x: np.round(x,2)*100)
vis.loc[:,'cluster'] = vis.loc[:,'cluster'].astype(int).apply(lambda x: 'C' + str(x)).astype('category')

unique_genres = ['Action','Adventure','All','Animation','Biography',
                 'Comedy','Crime','Documentary','Drama','Family',
                 'Fantasy','Film-Noir','Game-Show','History',
                 'Horror','Music','Musical','Mystery','News',
                 'Reality-TV','Romance','Sci-Fi','Short','Sport',
                 'Talk-Show','Thriller','War','Western']

cluster_labels = ['Catch-All 1','Cartoons & Music',
                  'Reality-TV, Talk & Game Shows',
                  'Documentaries','Catch-All 2','Comedies','Dramas']
cluster_labels2 = cluster_labels[:]
cluster_labels2.append('All')
cluster_labels2.sort()

cluster_centers = np.array([[ 2.63815789, -0.06592623],
                   [ 0.83339117,  0.33304976],
                   [-4.11374782, -1.16945493],
                   [ 2.56188663,  0.88783778],
                   [-3.72658814,  0.6036749 ],
                   [ 3.77430736, -3.89254365],
                   [ 2.94440904,  6.66672353]])

index_cmap1 = factor_cmap('cluster', palette=Set1[7], 
                          factors=['C0','C1','C2','C3','C4','C5','C6'])

index_cmap2 = factor_cmap('names', palette=Set1[7], 
                          factors=cluster_labels)

desc = Div(text=open(join(dirname(__file__), "description1.html")).read(), sizing_mode="stretch_width")

# Create Input controls
genreControl = Select(title="Genre", value="All", options=unique_genres)
titleControl = TextInput(title="Title of TV Series contains")
startYearControl = Slider(title="Start Year", start=1906, end=2024, value=1906, step=1)   
seasonNumberControl = Slider(title="Number of Seasons", start=1, end=78, value=1, step=1)
episodeNumberControl = Slider(title="Number of Episodes", start=1, end=12902, value=1, step=1)   
embeddingClusterControl = Select(title="Embedding Cluster", value="All", options=cluster_labels2)
entropyControl = Slider(title="Cluster Entropy (nats)",start=0, end=1.07, value=0, step=0.01)


# Create Column Data Source to be used by plot
source = ColumnDataSource(data=dict(x=[], y=[],  
                                    originalTitle=[], startYear=[], 
                                    seasonNumber=[],
                                    cluster=[],genres=[], 
                                    c0_prob=[], c1_prob=[], c2_prob=[], 
                                    c3_prob=[], c4_prob=[], c5_prob=[],
                                    c6_prob=[], entropy=[]))

TOOLTIPS = [
        ('Title','@originalTitle'),
        ('Start Year','@startYear'),
        ('Number of Seasons','@seasonNumber'),
        ('Genres','@genres'),
        ('Catch-All 1 (%)','@c0_prob'),
        ('Cartoons & Music (%)','@c1_prob'),
        ('Reality-TV, Talk & Game Shows (%)','@c2_prob'),
        ('Documentaries (%)','@c3_prob'),
        ('Catch-All 2 (%)','@c4_prob'),
        ('Comedies (%)','@c5_prob'),
        ('Dramas (%)','@c6_prob')
        ]


p = figure(plot_height=700, plot_width=1100, title="", sizing_mode='scale_both', tooltips=TOOLTIPS)
p.scatter(x='x', y='y', source=source, size=8, fill_color=index_cmap1, line_color=None, alpha=0.8)

labelSource = ColumnDataSource(data=dict(dim1=cluster_centers[:,0],
                                         dim2=cluster_centers[:,1],
                                         names=cluster_labels))

labels = LabelSet(x = 'dim1', y='dim2', text='names',
                  source=labelSource, render_mode='canvas', text_font='helvetica',
                  text_font_size='12pt',
                  text_color=index_cmap2, background_fill_color='white')

p.add_layout(labels)

def select_series():
    genre_val = genreControl.value
    title_val = titleControl.value.strip()
    cluster_val = embeddingClusterControl.value
    selected = vis[(vis.startYear >= startYearControl.value) &
                   (vis.seasonNumber >= seasonNumberControl.value) & 
                   (vis.episodeNumber >= episodeNumberControl.value) &
                   (vis.entropy >=  entropyControl.value)]
    if (genre_val != "All"):
        selected = selected[selected.genre1.str.contains(genre_val) |
                            selected.genre2.str.contains(genre_val) |
                            selected.genre3.str.contains(genre_val)]
    if (title_val != ""):
        selected = selected[selected.originalTitle.str.contains(title_val)]
    if (cluster_val == 'Catch-All 1'):
        selected = selected[selected.cluster == 'C0']
    elif (cluster_val == 'Cartoons & Music'):
        selected = selected[selected.cluster == 'C1']
    elif (cluster_val == 'Reality-TV, Talk & Game Shows'):
        selected = selected[selected.cluster == 'C2']
    elif (cluster_val == 'Documentaries'):
        selected = selected[selected.cluster == 'C3']
    elif (cluster_val == 'Catch-All 2'):
        selected = selected[selected.cluster == 'C4']
    elif (cluster_val == 'Comedies'):
        selected = selected[selected.cluster == 'C5']
    elif (cluster_val == 'Dramas'):
        selected = selected[selected.cluster == 'C6']
    return selected


def update():
    df = select_series()
    p.title.text = 'TV Series Embedding 2D Visualizations: {} TV series selected'.format(len(df))
    p.title.text_font_size = '20pt'
    source.data = df
    source.data = dict(x=df['dim1'],
                       y=df['dim2'],
                       originalTitle=df['originalTitle'],
                       startYear=df['startYear'],
                       seasonNumber=df['seasonNumber'],
                       cluster=df['cluster'],
                       genres=df['genres'],
                       c0_prob=df['c0_prob'],
                       c1_prob=df['c1_prob'],
                       c2_prob=df['c2_prob'],
                       c3_prob=df['c3_prob'],
                       c4_prob=df['c4_prob'],
                       c5_prob=df['c5_prob'],
                       c6_prob=df['c6_prob'],
                       entropy=df['entropy'])
    
controls = [genreControl, titleControl, startYearControl, seasonNumberControl, 
            episodeNumberControl, embeddingClusterControl, entropyControl]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(desc,*controls, width=500, height=700)
inputs.sizing_mode = 'fixed'
l = layout([[inputs,p]],sizing_mode='scale_both')

update()

curdoc().add_root(l)
curdoc().title = 'TV Series 2D Visualization'  
