#/!bin/bash
pipi_url="https://localhost:5443/repository/${USER}-pypi-dev/simple/"
if [ $(id -u) -ne 0 ]; then
  echo "WARNING: As of 2025-04-20, it is not safe to install wml packages locally."
  wml_nexus.py pip3 install --trusted-host localhost --extra-index $pipi_url --upgrade $@
else
  wml_nexus.py uv pip install -p /usr/bin/python3 --system --break-system-packages --prerelease allow --allow-insecure-host localhost --index $pipi_url --index-strategy unsafe-best-match $@
fi
