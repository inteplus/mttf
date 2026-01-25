#/!bin/bash
pipi_url="https://localhost:5443/repository/${USER}-pypi-dev/simple/"
if [ $(id -u) -ne 0 ]; then
  echo "WARNING: As of 2026-01-24, you need to create a virtual environment (e.g. uv venv) before pip-installing wml packages."
  wml_nexus.py uv pip install --allow-insecure-host localhost --index $pipi_url --index-strategy unsafe-best-match --link-mode=copy $@
else
  wml_nexus.py uv pip install -p /usr/bin/python3 --system --break-system-packages --prerelease allow --allow-insecure-host localhost --index $pipi_url --index-strategy unsafe-best-match --link-mode=copy $@
fi
