# QuaRot_Re-Implementation_on_Llama_3.2_1Bit
Steps to generate FX graph and add nodes in it to rotate the weights offline

* Install uv using **pip install uv**
* Create a virtual environment using **python3 -m venv .venv** and activate virtual environment using **source .venv/bin/activate**
* Run **uv sync** to download and install the python packages locally
* Run **uv run graph_version.py** to create an exported version of model and insert nodes in the graph to perform offline rotations