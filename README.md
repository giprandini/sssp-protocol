# sssp-protocol
This repository contains all the necessary workflows and tools in order to run the SSSP (standard solid state pseudopotential) protocol for testing pseudopotentials.

#### How to use

- Install AiiDA (version 0.7) following the instructions in the documentation (https://aiida-core.readthedocs.io/en/v0.7.0/) and the plugin for Quantum ESPRESSO.
- Compile the Fortran77 subroutine for the Fermi energy with `efermi_f2py_make.sh` through f2py

#### Content

scripts:
- run_ssspworkflow.py                -->  input script to launch the full SsspWorkflow 
- sssp_convergence_plot.py           -->  script to generate the SSSP convergence pattern plots from a previously run SsspWorkflow
- sssp_eos_plot.py                   -->  script to generate the plots of the equations of states

#### Acknowledgements

If you use this software, please cite the following works:

SSSP: G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet and N. Marzari, npj Computational Materials 4, 72 (2018). 
WEB: http://materialscloud.org/sssp.

K. Lejaeghere et al., Science 351 (6280), 1415 (2016). 
DOI: 10.1126/science.aad300
WEB: http://molmod.ugent.be/deltacodesdft.
