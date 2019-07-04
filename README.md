# sssp-protocol
This repository contains all the necessary workflows and tools in order to run the SSSP (standard solid state pseudopotential) protocol for testing pseudopotentials.

#### Prerequisites

 * python 2.7
 * fortran compiler
 * Quantum Espresso 
 * aiida-core v0.7 (see [documentation](https://aiida-core.readthedocs.io/en/v0.7.0/) for instructions)
 * aiida-quantumespresso 

#### Installation

```
git clone https://github.com/giprandini/sssp-protocol
pip install sssp-protocol
```

- Compile the Fortran77 subroutine for the Fermi energy with `efermi_f2py_make.sh` through f2py

#### Usage example

We include a simple example to test the SsspWorkflow on elemental palladium and PAW pseudopotential from the PsLibrary 0.3.1. 
In order to run the example you need to:

- enter in the example directory and load the pseudopotential file 'Pd.pbe-n-kjpaw_psl.0.3.0.UPF' within an AiiDA pseudopotential family with name 'pslib.0.3.1_PBE_PAW'
- download and insert in the current directory the file 'Wien2k.txt' from the calcDelta package (available at https://molmod.ugent.be/deltacodesdft) having the all-electron data for computing the DeltaFactor from the equation of state
- run the AiiDA script 'load_aiida_data.py' to load and store in the database the necessary data for the example
- run the AiiDA input script 'run_sssp-example.py' to launch the SsspWorkflow
- when the SsspWorkflow is in FINISHED state, run the AiiDA scripts 'convergence-plot_example.py' to generate the SSSP convergence plot and 'eos-plot_example.py' to generate the plot of the equation of state


#### Content

scripts:
- run_ssspworkflow.py                -->  example of input script used to launch the full SsspWorkflow 
- sssp_convergence_plot.py           -->  script to generate the SSSP convergence pattern plots from a previously run SsspWorkflow
- sssp_eos_plot.py                   -->  script to generate the plots of the equations of states

#### Acknowledgements

If you use this software, please cite the following works:

SSSP: G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet and N. Marzari, npj Computational Materials 4, 72 (2018). 
WEB: http://materialscloud.org/sssp.

K. Lejaeghere et al., Science 351 (6280), 1415 (2016). 
DOI: 10.1126/science.aad300
WEB: http://molmod.ugent.be/deltacodesdft.
