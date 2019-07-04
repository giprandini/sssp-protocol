from __future__ import division, absolute_import, print_function

from numpy.distutils.core import Extension, setup

ef_ext = Extension(
        name = 'efermi_module',
        sources = ['sssp_tools/efermi.pyf', 'sssp_tools/efermi.f'],
)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sssp_protocol",
    version="0.0.1",
    author="Gianluca Prandini, Antimo Marrazzo, Ivano E. Castelli, Nicolas Mounet, Nicola Marzari",
    author_email="gianluca.prandini@epfl.ch",
    description="Workflows and tools for running the SSSP (standard solid state pseudopotential) protocol",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giprandini/sssp-protocol/",
    packages=['sssp_tools'],
    setup_requires=['numpy'],
    install_requires=[
        'aiida-core==0.7.0.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT and GNU licenses",
        "Operating System :: OS Independent",
    ],
    ext_modules = [ef_ext],
)
