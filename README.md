# sssp-protocol
This repository contains all the necessary workflows and tools in order to run the SSSP (standard solid state pseudopotential) protocol for testing pseudopotentials.

#### Prerequisites

 * python 2.7
 * fortran compiler
 * Quantum Espresso 5.4-6.2
 * aiida-core v0.7.0 (see [documentation](https://aiida-core.readthedocs.io/en/v0.7.0/) for instructions)
   - Copy `aiida.patch` to `aiida-core` directory and apply: `patch -p1 < aiida.patch`
   - Copy `sssp_tools/sssp.py` to `aiida/workflows`
 * aiida-quantumespresso 
   - Use `verdi code setup` in order to set up an AiiDA code for Quantum Espresso 

#### Installation

```
git clone https://github.com/giprandini/sssp-protocol
pip install sssp-protocol
```
 
- Compile the Fortran77 subroutine for the Fermi energy with `efermi_f2py_make.sh` through f2py

#### Usage example

This demonstrates how to run the SsspWorkflow on elemental palladium for a PAW pseudopotential from the PsLibrary 0.3.1.
pseudopotential from the PsLibrary 0.3.1. 

- Create a pseudopotential family from the palladium pseudopotential:
  ```
  cd example/
  verdi data upf uploadfamily pseudopotentials 'pslib.0.3.1_PBE_PAW' 'SSSP test'
  ```

- run the AiiDA script 'load_aiida_data.py' to load and store in the database the necessary data for the example
- Start the daemon via `verdi daemon start`
- Launch the full `SsspWorkflow` using:
  ```
  verdi run run_sssp-example.py <structure_pk> <pw-code-name> <ph-code-name> <wien2k-pk> <kpoints-pk>
  ```
- Check the when the SsspWorkflow is in FINISHED state, run the AiiDA scripts 'convergence-plot_example.py' to generate the SSSP convergence plot and 'eos-plot_example.py' to generate the plot of the equation of state

- Plotting results
  * `sssp_convergence_plot.py`: plot the convergence pattern from a previously run SsspWorkflow
  * `sssp_eos_plot.py`: plot the equations of state

Note: In order to compute the Delta Factor from the equation of state, this
example uses the all-electron data inside 'WIEN2k.txt' taken from the calcDelta
package v3.1 available at https://molmod.ugent.be/deltacodesdft.

#### Acknowledgements

If you use this software, please cite the following works:

SSSP: G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet and N. Marzari, npj Computational Materials 4, 72 (2018). 
WEB: http://materialscloud.org/sssp.

K. Lejaeghere et al., Science 351 (6280), 1415 (2016). 
DOI: 10.1126/science.aad300
WEB: http://molmod.ugent.be/deltacodesdft.
