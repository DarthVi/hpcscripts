#!/bin/bash

pip install --user pathlib
pip install --user numpy
pip install --user scipy
pip install --user sckit-learn
pip install --user pandas
pip install --user plotly
pip install --user streamlit

wget https://github.com/plotly/orca/releases/download/v1.3.1/orca-1.3.1.AppImage -O /usr/local/bin/orca
chmod +x /usr/local/bin/orca

pip install --user psutil requests