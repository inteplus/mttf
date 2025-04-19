#/!bin/bash
wml_nexus.py pip3 install --pre --trusted-host localhost --extra-index https://localhost:5443/repository/ml-py-repo/simple/ $@
