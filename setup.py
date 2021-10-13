#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:17:09 2021

@author: ivan
"""

import os
import sys
from setuptools import setup, find_packages

setup(
    name="TSED",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="1.0",
    author="Ivan José dos Reis Filho",
    author_email="ivanfilhoreis@gmail.com",
    description="Time-Series Enriched with Domain-specific terms",
    
    long_description_content_type="text/markdown",
    url="https://github.com/ivanfilhoreis/TSED",
    keywords="time-series text mining Enriched Series",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3.6',
)
