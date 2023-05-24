#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages
from mt.tf.mttf_version import version

setup(
    name="mttf",
    version=version,
    description="A package to detect and monkey-patch TensorFlow, for Minh-Tri Pham",
    author=["Minh-Tri Pham"],
    packages=find_namespace_packages(include=["mt.*"]),
    install_requires=[
        # 'tensorflow', 'tensorflow-cpu' or 'tensorflow-gpu'
        "mtbase>=4.2",  # just updating
        "mtnet",  # for some basic networking support
    ],
    scripts=[
        "scripts/wml_nexus.py",  # for accessing Winnow Nexus repo
        "scripts/wml_pipi.sh",  # for accessing Winnow Nexus repo
        "scripts/wml_twineu.sh",  # for accessing Winnow Nexus repo
        "scripts/twine_trusted.py",  # for the ability to disable ssl verification
    ],
    url="https://github.com/inteplus/mttf",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.tf/mt.tf.html",
        "Source Code": "https://github.com/inteplus/mttf",
    },
)
