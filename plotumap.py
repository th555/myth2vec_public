"""
Adapted from
https://gist.github.com/pmbaumgartner/adb33aa486b77ab58eb3df265393195d
by Peter Baumgartner
"""

import umap
import numpy as np

import plotly
import plotly.graph_objs as go


def plot_model(wv, min_count=100):
    """ Gensim api changed since this script was made, get vectors and labels as in
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
    """
    vectors = np.asarray(wv.vectors)
    words = np.asarray(wv.index_to_key)
    counts = np.asarray([wv.get_vecattr(word, 'count') for word in words])
    logcounts = np.log10(counts) # for colouring

    if min_count:
        vectors = vectors[counts>=min_count]
        words = words[counts>=min_count]
        logcounts = logcounts[counts>=min_count]
        counts = counts[counts>=min_count]

    #n_neighbors was default(15)
    embedding = umap.UMAP(n_neighbors=10).fit_transform(vectors)

    tooltips = []
    for word, count in zip(words, counts):
        tooltips.append(f'{word}, n={count}')

    trace = go.Scattergl(
        x = embedding[:,0],
        y = embedding[:,1],
        name = 'Embedding',
        mode = 'markers',
        
        marker = dict(
            color = logcounts,
            colorscale='Viridis',
            size = 6,
            line = dict(
                width = 0.5,
            ),
            opacity=0.75
        ),
        text=tooltips
    )

    layout = dict(title = 'Word2Vec Mythfic - 2D UMAP Embeddings',
                  yaxis = dict(zeroline = False),
                  xaxis = dict(zeroline = False),
                  hovermode = 'closest'
                 )

    fig = go.Figure(data=[trace], layout=layout)
    chart = plotly.offline.plot(fig, filename='w2v-umap.html')
