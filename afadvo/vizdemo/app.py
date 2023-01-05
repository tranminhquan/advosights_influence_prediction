# -*- coding: utf-8 -*-
import os
import dash
import sys

from demo import create_layout, demo_callbacks
from nn.models.csif import *
from nn.models.hiersoftmax import *

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))



# for the Local version, import local_layout and local_callbacks
# from local import local_layout, local_callbacks

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

server = app.server
app.layout = create_layout(app)
demo_callbacks(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
