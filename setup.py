import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sssp_protocol",
    version="0.0.1",
    author="Gianluca Prandini, Antimo Marrazzo, Ivano E. Castelli, Nicolas Mounet, Nicola Marzari",
    author_email="gianluca.prandini@epfl.ch",
    description="SSSP protocol repository",
    long_description="This repository contains all the necessary workflows and tools in order to run the SSSP (standard solid state pseudopotential) protocol for testing pseudopotentials",
    long_description_content_type="text/markdown",
    url="https://github.com/giprandini/sssp-protocol/",
    packages=['sssp_tools'], #setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT and GNU licenses",
        "Operating System :: OS Independent",
    ],
)
