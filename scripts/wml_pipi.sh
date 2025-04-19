#/!bin/bash
wml_nexus.py uv pip install --trusted-host localhost --extra-index https://localhost:5443/repository/ml-py-repo/simple/ $@
