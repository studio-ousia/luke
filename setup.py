# -*- coding: utf-8 -*-

from setuptools import setup, Extension

setup(
    name='luke',
    version='0.0.1',
    description='Language understanding with knowledge-based embeddings',
    author='Ikuya Yamada',
    packages=['luke'],
    include_package_data=True,
    entry_points=dict(console_scripts=['luke=luke.cli:cli']),
)
