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
        "mtbase>=3.7",  # to have from mt import tp
    ],
    url="https://github.com/inteplus/mttf",
    project_urls={
        "Documentation": "https://mtdoc.readthedocs.io/en/latest/mt.tf/mt.tf.html",
        "Source Code": "https://github.com/inteplus/mttf",
    },
)
