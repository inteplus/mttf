#/!bin/bash
if [ $(id -u) -ne 0 ]; then
  echo "WARNING: As of 2025-04-20, it is not safe to install wml packages locally."
  wml_nexus.py uv pip install --index https://localhost:5443/repository/ml-py-repo/simple/ --prerelease allow $@
else
  wml_nexus.py uv pip install -p /usr/bin/python3 --system --break-system-packages --prerelease allow --index https://localhost:5443/repository/ml-py-repo/simple/ $@
fi
