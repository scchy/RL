from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="toyRL",
    version="0.1.0",
    description="simple RL toy package",
    packages=find_packages(),
    install_requires=required
)
