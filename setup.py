#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mttf",
    description="A package to detect and monkey-patch TensorFlow, for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=[
        # 'tensorflow', 'tensorflow-cpu' or 'tensorflow-gpu'
        "mtbase>=4.26",  # just updating
        "mtnet>=0.2",  # for some basic networking support
    ],
    scripts=[
        "scripts/wml_nexus.py",  # for accessing Winnow Nexus repo
        "scripts/wml_pipi.sh",  # for accessing Winnow Nexus repo
        "scripts/wml_twineu.sh",  # for accessing Winnow Nexus repo
        "scripts/dmt_pipi.sh",  # for accessing Winnow Nexus repo, MT dev environment
        "scripts/dmt_twineu.sh",  # for accessing Winnow Nexus repo, MT dev environment
        "scripts/twine_trusted.py",  # for the ability to disable ssl verification
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
