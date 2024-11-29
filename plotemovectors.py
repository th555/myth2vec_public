import umap
import numpy as np

import plotly
import plotly.graph_objs as go


def plot_emo(vectors, labels, vectorlabels):
    embedding = umap.UMAP(n_neighbors=4).fit_transform(vectors)

    tooltips = []
    for label, vector in zip(labels, vectors):
        tooltips.append(f'{label}, {vector}')

    trace = go.Scattergl(
        x = embedding[:,0],
        y = embedding[:,1],
        name = 'Embedding',
        mode = 'markers+text',
        
        marker = dict(
            size = 6,
            line = dict(
                width = 0.5,
            ),
            opacity=0.75
        ),
        text=labels,
        hovertext=tooltips,
    )

    layout = dict(title = 'Word2Vec Mythfic - 2D UMAP Embeddings of Emotion Vectors',
                  yaxis = dict(zeroline = False),
                  xaxis = dict(zeroline = False),
                  hovermode = 'closest'
                 )

    fig = go.Figure(data=[trace], layout=layout)
    chart = plotly.offline.plot(fig, filename='w2v-emotion_vectors.html')
