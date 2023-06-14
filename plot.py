import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from network import Network

points = np.random.uniform(0, 1, size=(10000, 2))
print(points)

close = []
far = []
nn = Network()
for point in points:
    if nn.eval(point): close.append(point)
    else: far.append(point)
close = np.array(close)
far = np.array(far)
fig = go.Figure()
fig.add_trace(go.Scatter(x=close[:, 0], y=close[:, 1],  mode="markers", marker_color="green"))
fig.add_trace(go.Scatter(x=far[:, 0], y=far[:, 1],  mode="markers", marker_color="red"))
fig.add_shape(type="circle",
    xref="x", yref="y",
    x0=min(close[:, 0]), y0=min(close[:, 1]),
    x1=max(close[:, 0]), y1=max(close[:, 1]),
    opacity=0.2,
    fillcolor="lightseagreen",
    line_color="lightseagreen",
)
fig.show()
