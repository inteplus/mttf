#/!bin/bash
wml_nexus.py uv pip install --python-version 3.8 --allow-insecure-host localhost -i https://localhost:5443/repository/ml-py-repo/simple/ $@
