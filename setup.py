#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mttf",
    description="A package to detect and monkey-patch TensorFlow and Keras, for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=[
        # 'tensorflow', 'tensorflow-cpu' or 'tensorflow-gpu'
        "pyyaml",
        "mtbase>=4.32.3",  # to rely on uv
        "mtnet>=0.3.2",  # just updating
    ],
    scripts=[
        "scripts/wml_nexus.py",  # for accessing Winnow Nexus repo
        "scripts/pipi.sh",  # for pip-installing packages using uv
        "scripts/wml_pipi.sh",  # for accessing Winnow Nexus repo
        "scripts/wml_twineu.sh",  # for accessing Winnow Nexus repo
        "scripts/dmt_pipi.sh",  # for accessing Winnow Nexus repo, MT dev environment
        "scripts/dmt_twineu.sh",  # for accessing Winnow Nexus repo, MT dev environment
    ],
    url="https://github.com/inteplus/mttf",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.tf/mt.tf.html",
        "Source Code": "https://github.com/inteplus/mttf",
    },
    setup_requires=["setuptools-git-versioning<2"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)
