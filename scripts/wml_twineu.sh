#/!bin/bash
CURL_CA_BUNDLE="" wml_nexus.py twine upload --repository-url https://localhost:5443/repository/ml-py-repo/ -u minhtri -p Winnow2019python $@
