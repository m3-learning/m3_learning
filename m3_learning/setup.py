from setuptools import setup, find_packages
import os

with open("src/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="m3_learning",
    version="0.0.16",
    packages=find_packages(where="src"),
    url="https://github.com/jagar2/m3_learning.git",
    install_requires=requirements,
    license=" BSD-3-Clause",
    author="Joshua C. Agar",
    author_email="jca92@drexel.edu",
    description="Tutorials, Projects, and datasets from the M3-learning research group",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
)
