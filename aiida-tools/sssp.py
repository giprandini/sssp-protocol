# -*- coding: utf-8 -*-
from __future__ import division
from aiida.common import aiidalogger
from aiida.orm.workflow import Workflow
from aiida.orm.utils import load_node, load_workflow
from aiida.orm import  Group, CalculationFactory, DataFactory
from sssp-protocol.aiida-tools.quantumespresso.pw import PwWorkflow
from sssp-protocol.aiida-tools.quantumespresso.phonondispersion import PhonondispersionWorkflow
from sssp-protocol.aiida-tools.quantumespresso.helpers import get_pw_wfs_with_parameters, get_phonondispersion_wfs_with_parameters
from aiida.common.exceptions import NotExistent
from aiida.orm.data.singlefile import SinglefileData
from sssp-protocol.aiida-tools import sssp_utils
from sssp-protocol.aiida-tools.sssp_utils import objects_are_equal
# from aiida.backends.djsite.db import models


__version__ = "1.0"
__authors__ = "Gianluca Prandini, Antimo Marrazzo, Nicolas Mounet"


UpfData = DataFactory('upf')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')
BandsData = DataFactory('array.bands')
YamboCalc = CalculationFactory('yambo')
PwCalculation = CalculationFactory('quantumespresso.pw')
logger = aiidalogger.getChild('SsspWorkflow')


    
class SsspWorkflow(Workflow):
    """
    
    { 'delta_' + any parameter to be used in the pw workflow for the delta calculation
     
      'bands_' + any parameter to be used in the pw workflow for the band structure calculation
      
      'stress_' + any parameter to be used in the pw workflow for the scf stress calculation
            
      'phonon_' + any parameter to be used in the phonondispersion workflow for the phonon calculation
    
      'cohesive_' + any parameter to be used in the pw workflow for the cohesive energy calculation (bulk and gas)
      
      'pseudo_family': pseudo family name,
      'label_dict': abbreviations for the pseudo_family names,
      'results_group_name': group name where to save ParameterData with the results of the convergence tests,
      
      'input': {'cutoffs': list of cutoffs (in Rydberg),
                'dual': dual,
                'compute_delta': (Boolean), launch delta calculation,
                'compute_cohesive': (Boolean), launch cohesive energy convergence,
                'compute_stress': (Boolean), launch stress convergence,
                'compute_bands': (Boolean), launch band structure convergence,
                'compute_phonon': (Boolean), launch phonon convergence,
                }
                
      'structure': AiiDA structure (structure used for convergence test) 
                  (taken from calcDelta package and reduced to primitive cells with spglib)
      'delta_structure': AiiDA structure (structure used for delta calculation) 
                    (taken from calcDelta package 'CIFs.tar.gz')
      'cohesive_gas_structures_group_name': the name of the group where to find the structures of the isolated atom     
      
     }
    
    step order:
        start
        run_cohesive_and_delta:
            delta (pw wf)
            cohesive_gas (pw wf)
            cohesive_bulk (pw wf) 
        run_stress:
            stress (pw wf)
        run_bands_and_phonon:
            bands (pw wf)
            phonon (phonondispersion wf)
        final_step:
            use inline calcs:
                get_compute_Delta_and_Birch_Murnaghan_EOS_results for delta evaluation,
                get_bands_distance_results for bands distance evaluation,
                get_bands_distance_info_results for storing bands distances inside a single ParameterData,
                get_build_info_results for storing the results of the convergence tests inside ParameterData (one for each cutoff)
    """
    
    _default_delta_points = 7
    _default_delta_step = 0.02
    
    def __init__(self, **kwargs):
        super(SsspWorkflow, self).__init__(**kwargs)
        
    @Workflow.step
    def start(self):
        """
        Check the input parameters of the workflow
        """
        main_params = self.get_parameters()
        self.append_to_report("Starting the SsspWorkflow ...")
        self.append_to_report("Checking SSSP input parameters")
               
        # Check the params to see which workflow step to launch       
        launch_delta = main_params['input']['compute_delta']
        launch_cohesive = main_params['input']['compute_cohesive']  
        launch_stress = main_params['input']['compute_stress']
        launch_bands = main_params['input']['compute_bands']
        launch_phonon = main_params['input']['compute_phonon']  
        
        # Check that the launch_* variables are of boolean type. Not needed, it should be checked in the input.
        if ( type(launch_delta) or type(launch_cohesive) or type(launch_stress) or type(launch_bands) or \
              type(launch_phonon) ) != bool:
            raise TypeError('Input parameters compute_* must be boolean')
        
        # With this logic the ordering of the 'if' is important!
        if (launch_delta or launch_cohesive):
            self.next(self.run_cohesive_and_delta)
        elif launch_stress:    
            self.next(self.run_stress)
        elif (launch_bands or launch_phonon):
            self.next(self.run_bands_and_phonon)
        else:    
            self.next(self.final_step)

    @Workflow.step
    def run_cohesive_and_delta(self):
        """
        Launch pw sub-workflows for delta and cohesive energy calculation
        """
        main_params = self.get_parameters()

        # """ Delta calculation """
        launch_delta = main_params['input']['compute_delta']
        if launch_delta:
            # take the parameters needed for the delta calculation (PwWorkflow)
            delta_pw_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('delta_pw_'):
                    new_k = k[9:] # remove 'delta_pw_' from the key name
                    delta_pw_params[new_k] = v
                                
            delta_pw_params['pseudo_family'] = main_params['pseudo_family']
            delta_pw_params['codename'] = main_params['pw_codename']
            delta_pw_params['structure'] = main_params['delta_structure']
            try:
                delta_pw_params['input']['relaxation_scheme'] = 'scf'
            except KeyError:
                delta_pw_params['input'] = {'relaxation_scheme': 'scf'}
            
            if 'kpoints' not in delta_pw_params:
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(main_params['delta_parameters']['kpoints_mesh'])
                kpoints.store()
                delta_pw_params['kpoints'] = kpoints
            
            structure = delta_pw_params['structure']
            delta_wf_pks_already_computed =[]
            for i in xrange( int(round(-self._default_delta_points/2+1)), int(self._default_delta_points/2+1) ):
                p_scale = ParameterData(dict={'scaling_ratio':1+i*self._default_delta_step})
                result_dict = sssp_utils.get_rescale_results(parameters=p_scale, structure=structure)
                rescaled_structure = result_dict['rescaled_structure']
                delta_pw_params['structure'] = rescaled_structure
                rescaled_volume = rescaled_structure.get_cell_volume()
                
                # setup of the magnetic elements: O, Mn, Cr, Fe, Co, Ni, RE nitrides
                if rescaled_structure.get_ase().get_chemical_symbols()[0] == 'O':
                    delta_pw_params['parameters']['SYSTEM']['nspin'] = 2
                    delta_pw_params['parameters']['SYSTEM']['starting_magnetization'] = {'O1': 0.5, 'O2': 0.5, 'O3': -0.5, 'O4': -0.5}
                if rescaled_structure.get_ase().get_chemical_symbols()[0] == 'Mn':
                    delta_pw_params['parameters']['SYSTEM']['nspin'] = 2
                    delta_pw_params['parameters']['SYSTEM']['starting_magnetization'] = {'Mn1': 0.5, 'Mn2': -0.3, 'Mn3': 0.5, 'Mn4': -0.3}
                if rescaled_structure.get_ase().get_chemical_symbols()[0] == 'Cr':
                    delta_pw_params['parameters']['SYSTEM']['nspin'] = 2
                    delta_pw_params['parameters']['SYSTEM']['starting_magnetization'] = {'Cr1': 0.5, 'Cr2': -0.5}
                if rescaled_structure.get_ase().get_chemical_symbols()[0] in ['Fe','Co','Ni']:
                    delta_pw_params['parameters']['SYSTEM']['nspin'] = 2
                    delta_pw_params['parameters']['SYSTEM']['starting_magnetization(1)'] = 0.2
                if rescaled_structure.get_ase().get_chemical_symbols()[0] in ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']:
                    delta_pw_params['parameters']['SYSTEM']['nspin'] = 2
                    # I put the starting_magnetization on the RE atom
                    delta_pw_params['parameters']['SYSTEM']['starting_magnetization'] = {'N': 0.0, 
                                    '{}'.format(rescaled_structure.get_ase().get_chemical_symbols()[0]): 0.2} 

#                     if rescaled_structure.get_ase().get_chemical_symbols()[0] > 'N':
#                         delta_pw_params['parameters']['SYSTEM']['starting_magnetization(2)'] = 0.2
#                     else:
#                         delta_pw_params['parameters']['SYSTEM']['starting_magnetization(1)'] = 0.2
                        
                
                previous_delta_wfs = get_pw_wfs_with_parameters(delta_pw_params, also_bands=False)
                previous_delta_wfs_states = [wf.get_state() for wf in previous_delta_wfs]
                if previous_delta_wfs and all(state == 'ERROR' for state in previous_delta_wfs_states):
                    self.append_to_report("The delta calculation for volume = {} Angstrom^3 was already launched but it is in ERROR"
                                          " state. Launching again. Previous failed PwWorkflows pks"
                                          " are: {}".format(rescaled_volume, [_.pk for _ in previous_delta_wfs]))
                    previous_delta_wfs = []
                if not previous_delta_wfs:
                    wf = PwWorkflow(params=delta_pw_params)
                    self.attach_workflow(wf)
                    wf.start()
                    wf.add_attribute('delta_calculation',True)
                    wf.add_attribute('cohesive_bulk_calculation',False)
                    wf.add_attribute('cohesive_gas_calculation',False)
                    self.append_to_report("Launch PwWorkflow (pk={}) for delta"
                           " calculation with volume={} Angstrom^3".format(wf.pk, rescaled_volume))
                    self.append_to_report("PwWorkflow Parameters: {}".format(wf.get_parameters()))
                else:
                    # 'build_info_results' would fail if here I don't take only a single wf every rescaled_structure
                    # For the other tests instead it works without doing it: see how results_dict[cutoff] is built (it overwrites if there are multiple data for the same cutoff).
                    if len(previous_delta_wfs)>1:
                        self.append_to_report('WARNING! Found more than one previous PwWorkflow for the delta'
                            ' calculation. I will take the last one.')
                        previous_delta_wfs = [previous_delta_wfs[-1]]
                    self.append_to_report("The delta calculation for volume={} Angstrom^3 was already done."
                           " Previous PwWorkflows pks are: {}".format(rescaled_volume, [_.pk for _ in previous_delta_wfs]))
                    delta_wf_pks_already_computed.extend([_.pk for _ in previous_delta_wfs if _.get_state() != 'ERROR'])
            self.add_attribute('delta_wf_pks_already_computed',delta_wf_pks_already_computed) 
                      
        # """ Cohesive energy calculation """
        # TODO: It does not work now for the rare-earths and for F (composite structures RE-N and SiF4) 
        launch_cohesive = main_params['input']['compute_cohesive']
        if launch_cohesive:
            # take the parameters needed for the cohesive energy calculation (PwWorkflow)
            cohesive_pw_bulk_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('cohesive_pw_bulk_'):
                    new_k = k[17:] # remove 'cohesive_pw_bulk_' from the key name
                    cohesive_pw_bulk_params[new_k] = v
                    
            cohesive_pw_bulk_params['pseudo_family'] = main_params['pseudo_family']
            cohesive_pw_bulk_params['codename'] = main_params['pw_codename']
            cohesive_pw_bulk_params['structure'] = main_params['structure']
            cohesive_pw_bulk_params['input']['relaxation_scheme'] = 'scf'
            
            if 'kpoints' not in cohesive_pw_bulk_params:
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(main_params['cohesive_bulk_parameters']['kpoints_mesh'])
                kpoints.store()
                cohesive_pw_bulk_params['kpoints'] = kpoints
            
            cohesive_pw_gas_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('cohesive_pw_gas_'):
                    new_k = k[16:] # remove 'cohesive_pw_gas_' from the key name
                    cohesive_pw_gas_params[new_k] = v
                    
            cohesive_pw_gas_params['pseudo_family'] = main_params['pseudo_family']
            cohesive_pw_gas_params['codename'] = main_params['pw_codename']
            cohesive_pw_gas_params['input']['relaxation_scheme'] = 'scf'
            
            try:
                group_gas_structures = Group.get_from_string('{}'.format(main_params['cohesive_gas_structures_group_name']))
            except NotExistent:
                raise Exception("Group with name '{}' for 'cohesive_gas_structures_group_name' does not"
                                " exist!".format(main_params['cohesive_gas_structures_group_name']))
            # cohesive_pw_gas_params['structure'] = main_params['cohesive_gas_structure']
            
            # Gamma point (I think this should never be changed)
            kpoints = KpointsData()
            kpoints.set_kpoints_mesh([1,1,1])
            kpoints.store()
            cohesive_pw_gas_params['kpoints'] = kpoints
            
            # Converge test w.r.t. the wfc cutoff
            cutoffs = main_params['input']['cutoffs']
            dual = main_params['input']['dual']

            # Add possibility to give in input parameters_data with the 'parameters_energy_bulk' of the info_files
            # in order to calculate the isolated atoms but not the bulk
            # Maybe to delete/improve in the future
            only_isolated = False
            try:
                parameters_data = main_params['input']['parameters_data']
                if len(cutoffs)>1:
                    raise Exception('The input parameter "parameters_data" can be used only for a single cutoff')
                parameters_energy_bulk = load_node(parameters_data['parameters_energy_bulk'])
                if isinstance(parameters_energy_bulk, ParameterData) and isinstance(parameters_energy_bulk.get_dict()['energy'],float): 
                    only_isolated = True
                for k,v in parameters_data.iteritems():
                    if k.startswith('parameters_energy_gas_'):
                        only_isolated = False
                        break
                if only_isolated:
                    self.append_to_report("The cohesive energy (bulk) calculation for ecut={} Ry is already included "
                                          "in the 'parameters_data['parameters_energy_bulk']' (pk = {})"
                                          "".format(cutoffs[0], parameters_energy_bulk.pk))
            except KeyError:
                pass    

            
            cohesive_bulk_wf_pks_already_computed = []
            cohesive_gas_wf_pks_already_computed = []
            for cutoff in cutoffs:    
                
                # Bulk
                if not only_isolated:
                    cohesive_pw_bulk_params['parameters']['SYSTEM']['ecutwfc'] = cutoff
                    cohesive_pw_bulk_params['parameters']['SYSTEM']['ecutrho'] = cutoff*dual
                    
                    previous_cohesive_bulk_wfs = get_pw_wfs_with_parameters(cohesive_pw_bulk_params, also_bands=False)
                    previous_cohesive_bulk_wfs_states = [wf.get_state() for wf in previous_cohesive_bulk_wfs]
                    if previous_cohesive_bulk_wfs and all(state == 'ERROR' for state in previous_cohesive_bulk_wfs_states):
                        self.append_to_report("The cohesive energy (bulk) calculation for ecut={} Ry was already launched but it is in ERROR state."
                                              " Launching again. Previous failed PwWorkflows pks"
                                              " are: {}".format(cutoff, [_.pk for _ in previous_cohesive_bulk_wfs]))
                        previous_cohesive_bulk_wfs = []
                    if not previous_cohesive_bulk_wfs:
                        wf = PwWorkflow(params=cohesive_pw_bulk_params)
                        self.attach_workflow(wf)
                        wf.start()
                        wf.add_attribute('delta_calculation',False)
                        wf.add_attribute('cohesive_bulk_calculation',True)
                        wf.add_attribute('cohesive_gas_calculation',False)
                        self.append_to_report("Launch PwWorkflow (pk={}) for cohesive energy (bulk)"
                               " calculation with ecut={} Ry".format(wf.pk, cutoff))
                        self.append_to_report("PwWorkflow Parameters: {}".format(wf.get_parameters()))
                    else:
                        self.append_to_report("The cohesive energy (bulk) calculation for ecut={} Ry was already done."
                               " Previous PwWorkflows pks are: {}".format(cutoff, [_.pk for _ in previous_cohesive_bulk_wfs]))
                        cohesive_bulk_wf_pks_already_computed.extend([_.pk for _ in previous_cohesive_bulk_wfs if _.get_state() != 'ERROR'])
                
                # Gas
                cohesive_pw_gas_params['parameters']['SYSTEM']['ecutwfc'] = cutoff
                cohesive_pw_gas_params['parameters']['SYSTEM']['ecutrho'] = cutoff*dual
                
                for kind in cohesive_pw_bulk_params['structure'].kinds:
                    element = kind.symbol
                    structs = [s for s in StructureData.query(dbgroups=group_gas_structures.pk)
                                        if s.get_formula(mode='hill_compact')==element]
                    if len(structs)!=1:
                        self.append_to_report("ERROR: group {} (pk = {}) does not contain "
                                              "the {} structure for cohesive energy (gas) calculation or contains too "
                                              "many of them".format(
                                              group_gas_structures.name, group_gas_structures.pk,element))
                        raise ValueError("ERROR: cannot get structure for isolated atom {}".format(element))
                    cohesive_pw_gas_params['structure'] = structs[0]
                                   
                    previous_cohesive_gas_wfs = get_pw_wfs_with_parameters(cohesive_pw_gas_params, also_bands=False)
                    previous_cohesive_gas_wfs_states = [wf.get_state() for wf in previous_cohesive_gas_wfs]
                    if previous_cohesive_gas_wfs and all(state == 'ERROR' for state in previous_cohesive_gas_wfs_states):
                        self.append_to_report("The cohesive energy (gas) calculation for element={} and ecut={} Ry was"
                                              " already launched but it is in ERROR state."
                                              " Launching again. Previous failed PwWorkflows pks"
                                              " are: {}".format(element, cutoff, [_.pk for _ in previous_cohesive_gas_wfs]))
                        previous_cohesive_gas_wfs = []
                    if not previous_cohesive_gas_wfs:
                        wf = PwWorkflow(params=cohesive_pw_gas_params)
                        self.attach_workflow(wf)
                        wf.start()
                        wf.add_attribute('delta_calculation',False)
                        wf.add_attribute('cohesive_bulk_calculation',False)
                        wf.add_attribute('cohesive_gas_calculation',True)
                        self.append_to_report("Launch PwWorkflow (pk= {}) for cohesive energy (gas)"
                               " calculation for element={} with ecut={} Ry".format(wf.pk, element, cutoff))
                        self.append_to_report("PwWorkflow Parameters: {}".format(wf.get_parameters()))
                    else:
                        self.append_to_report("The cohesive energy (gas) calculation for element={} and ecut={} Ry was already done."
                               " Previous PwWorkflows pks are: {}".format(element, cutoff, [_.pk for _ in previous_cohesive_gas_wfs]))
                        cohesive_gas_wf_pks_already_computed.extend([_.pk for _ in previous_cohesive_gas_wfs if _.get_state() != 'ERROR'])  
        
            self.add_attribute('cohesive_bulk_wf_pks_already_computed',cohesive_bulk_wf_pks_already_computed)
            self.add_attribute('cohesive_gas_wf_pks_already_computed',cohesive_gas_wf_pks_already_computed)
            
        self.next(self.run_stress)

    @Workflow.step
    def run_stress(self):
        """
        Launch pw sub-workflows for stress calculation
        """
        main_params = self.get_parameters()
        
        # """ Stress calculation """
        launch_stress = main_params['input']['compute_stress']
        if launch_stress:
            stress_pw_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('stress_pw_'):
                    new_k = k[10:] # remove 'stress_pw_' from the key name
                    stress_pw_params[new_k] = v
        
            stress_pw_params['pseudo_family'] = main_params['pseudo_family']
            stress_pw_params['codename'] = main_params['pw_codename']
            stress_pw_params['structure'] = main_params['structure']
            try:
                stress_pw_params['input']['relaxation_scheme'] = 'scf'
            except KeyError:
                stress_pw_params['input'] = {'relaxation_scheme': 'scf'}
            
            if 'kpoints' not in stress_pw_params:
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(main_params['stress_parameters']['kpoints_mesh'])
                kpoints.store()
                stress_pw_params['kpoints'] = kpoints
            
            # Converge test w.r.t. the wfc cutoff
            cutoffs = main_params['input']['cutoffs']
            dual = main_params['input']['dual']
            stress_wf_pks_already_computed = []
            for cutoff in cutoffs: 
                stress_pw_params['parameters']['SYSTEM']['ecutwfc'] = cutoff
                stress_pw_params['parameters']['SYSTEM']['ecutrho'] = cutoff*dual
                
                previous_stress_wfs = get_pw_wfs_with_parameters(stress_pw_params, also_bands=False)
                previous_stress_wfs_states = [wf.get_state() for wf in previous_stress_wfs]
                if previous_stress_wfs and all(state == 'ERROR' for state in previous_stress_wfs_states):
                    self.append_to_report("The stress calculation for ecut={} Ry was already launched but it is in ERROR state."
                                          " Launching again. Previous failed PwWorkflows pks"
                                          " are: {}".format(cutoff, [_.pk for _ in previous_stress_wfs]))
                    previous_stress_wfs = []
                if not previous_stress_wfs:
                    wf = PwWorkflow(params=stress_pw_params)
                    self.attach_workflow(wf)
                    wf.start()
                    self.append_to_report("Launch PwWorkflow (pk={}) for stress"
                           " calculation with ecut={} Ry".format(wf.pk, cutoff))
                    self.append_to_report("PwWorkflow Parameters: {}".format(wf.get_parameters()))
                else:
                    self.append_to_report("The stress calculation for ecut={} Ry was already done."
                           " Previous PwWorkflows pks are: {}".format(cutoff, [_.pk for _ in previous_stress_wfs]))
                    stress_wf_pks_already_computed.extend([_.pk for _ in previous_stress_wfs if _.get_state() != 'ERROR'])
            self.add_attribute('stress_wf_pks_already_computed',stress_wf_pks_already_computed)
         
        self.next(self.run_bands_and_phonon)

    @Workflow.step
    def run_bands_and_phonon(self):
        """
        Launch pw sub-workflows for band structures calculation and phonondispersion sub-workflows for 
        phonon calculation
        """
        main_params = self.get_parameters()
        
        # """ Band calculation """
        launch_bands = main_params['input']['compute_bands']
        if launch_bands:

            structure = main_params['structure']
           
            bands_pw_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('bands_pw_'):
                    new_k = k[9:] # remove 'bands_pw_' from the key name
                    bands_pw_params[new_k] = v

            bands_pw_params['pseudo_family'] = main_params['pseudo_family']
            bands_pw_params['codename'] = main_params['pw_codename']
            bands_pw_params['structure'] = main_params['structure']
            try:
                bands_pw_params['input']['relaxation_scheme'] = 'scf'
            except KeyError:
                bands_pw_params['input'] = {'relaxation_scheme': 'scf'}
            
            if 'kpoints' not in bands_pw_params:
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(main_params['bands_parameters']['kpoints_mesh'])
                kpoints.store()
                bands_pw_params['kpoints'] = kpoints
            
            # Converge test w.r.t. the wfc cutoff
            try:
                cutoffs = main_params['bands_parameters']['cutoffs']
            except KeyError:
                cutoffs = main_params['input']['cutoffs']
            dual = main_params['input']['dual']
            
            band_wf_pks_already_computed =[]
            for cutoff in cutoffs: 
                bands_pw_params['parameters']['SYSTEM']['ecutwfc'] = cutoff
                bands_pw_params['parameters']['SYSTEM']['ecutrho'] = cutoff*dual
                
                previous_bands_wfs = get_pw_wfs_with_parameters(bands_pw_params, also_bands=True)
                previous_bands_wfs_states = [wf.get_state() for wf in previous_bands_wfs]
                if previous_bands_wfs and all(state == 'ERROR' for state in previous_bands_wfs_states):
                    self.append_to_report("The band calculation for ecut={} Ry was already launched but it is in ERROR state."
                                          " Launching again. Previous failed PwWorkflows pks"
                                          " are: {}".format(cutoff, [_.pk for _ in previous_bands_wfs]))
                    previous_bands_wfs = []
                    
                if not previous_bands_wfs:
                    try:
                        noncolin_flag=bands_pw_params['parameters']['SYSTEM']['noncolin']
                    except KeyError:
                        noncolin_flag=False
                    if noncolin_flag is True:
                        self.append_to_report('Non-collinear calculation mode.')
                    wf = PwWorkflow(params=bands_pw_params)
                    self.attach_workflow(wf)
                    wf.start()
                    #wf.add_attribute('cutoff',cutoff)
                    self.append_to_report("Launch PwWorkflow (pk= {}) for band"
                           " calculation with ecut={} Ry".format(wf.pk, cutoff))
                    self.append_to_report("PwWorkflow Parameters: {}".format(wf.get_parameters()))
                else:
                    self.append_to_report("The band calculation for ecut={} Ry was already done."
                           " Previous PwWorkflows pks are: {}".format(cutoff, [_.pk for _ in previous_bands_wfs]))
                    band_wf_pks_already_computed.extend([_.pk for _ in previous_bands_wfs if _.get_state() != 'ERROR'])
            self.add_attribute('band_wf_pks_already_computed',band_wf_pks_already_computed)
        
        # """ Phonon calculation """
        launch_phonon = main_params['input']['compute_phonon'] 
        if launch_phonon:
            phonon_params = {}
            for k,v in main_params.iteritems():
                if k.startswith('phonon_'):
                    new_k = k[7:] # remove 'phonon_' from the key name
                    phonon_params[new_k] = v
            
            phonon_params['pseudo_family'] = main_params['pseudo_family']
            phonon_params['pw_codename'] = main_params['pw_codename']
            phonon_params['ph_codename'] = main_params['ph_codename']
            phonon_params['structure'] = main_params['structure']
            phonon_params['pw_input']['relaxation_scheme'] = 'scf'
            
            try:
                params = phonon_params.pop('parameters')
            except KeyError:
                pass
            try:
                kpoints_mesh = params.pop('kpoints_mesh')
            except KeyError:
                pass
            try:
                qpoint_crystal = params.pop('qpoint')
            except KeyError:
                pass
            
            structure = phonon_params['structure']
            if 'pw_kpoints' not in phonon_params:
                kpoints = KpointsData()
                kpoints.set_kpoints_mesh(kpoints_mesh)
                kpoints.store()
                phonon_params['pw_kpoints'] = kpoints
            if 'ph_qpoints' not in phonon_params:
                qpoints = KpointsData()
                qpoints.set_cell(structure.cell)
                qpoints.set_kpoints([qpoint_crystal], cartesian=False)
                qpoints.store()
                phonon_params['ph_qpoints'] = qpoints
            
            if structure.get_ase().get_chemical_symbols()[0] == 'O':
                phonon_params['ph_parameters']['INPUTPH']['alpha_mix(1)'] = 0.07
            
            # Converge test w.r.t. the wfc cutoff
            cutoffs = main_params['input']['cutoffs']
            dual = main_params['input']['dual']
            
            phonon_wf_pks_already_computed =[]
            for cutoff in cutoffs: 
                phonon_params['pw_parameters']['SYSTEM']['ecutwfc'] = cutoff
                phonon_params['pw_parameters']['SYSTEM']['ecutrho'] = cutoff*dual
                
                previous_phonondisp_wfs = get_phonondispersion_wfs_with_parameters(phonon_params)
                try: # check if the phonon calculation was already done
                    previous_phonon_wfs = previous_phonondisp_wfs['Phonondispersion']
                    previous_phonon_wfs_states = [wf.get_state() for wf in previous_phonon_wfs]
                    if previous_phonon_wfs_states and all(state == 'ERROR' for state in previous_phonon_wfs_states):
                        self.append_to_report("The phonon calculation for ecut={} Ry was already launched but it is "
                                              "in ERROR state. Launching again. Previous failed PhonondispersionWorkflow "
                                              "pks are: {}".format(cutoff, [_.pk for _ in previous_phonon_wfs]))
                        previous_phonondisp_wfs = {}
                    else:
                        self.append_to_report("The phonon calculation for ecut={} Ry was already done."
                           " Previous PhonondisperionWorkflow pks are: {}".format(cutoff, [_.pk for _ in previous_phonon_wfs]))
                        phonon_wf_pks_already_computed.extend([_.pk for _ in previous_phonon_wfs if _.get_state() != 'ERROR'])
                except KeyError: # if the key 'Phonondispersion' is not present in previous_phonondisp_wfs   
                    pass
                
                try: # try to start the phonon calculation from a previous pw_calculation
                    previous_pw_wfs = previous_phonondisp_wfs['Pw']
                    previous_pw_wfs_states = [wf.get_state() for wf in previous_pw_wfs]
                    if previous_pw_wfs_states and all(state == 'ERROR' for state in previous_pw_wfs_states):
                        self.append_to_report("The 'pw_calculation' of phonon for ecut={} Ry was already launched but it "
                                              "is in ERROR state. Launching again. Previous failed PwWorkflows pks"
                                              " are: {}".format(cutoff, [_.pk for _ in previous_pw_wfs]))
                    else:
                        if len(previous_pw_wfs)>1:
                            self.append_to_report("WARNING! Found more than one previous pw_calculation for phonon."
                                                 " I take the last one.")
                        pw_calculation = previous_pw_wfs[-1].get_result('pw_calculation')
                        phonon_params['pw_calculation'] = load_node(pw_calculation.pk)
                        wf = PhonondispersionWorkflow(params=phonon_params)
                        self.attach_workflow(wf)
                        wf.start()
                        self.append_to_report("Launch PhonondispersionWorkflow (pk= {}) for phonon"
                           " calculation with ecut={} Ry from previous "
                           "'pw_calculation' (pk= {})".format(wf.pk, cutoff, pw_calculation.pk))
                        self.append_to_report("PhonondispersionWorkflow Parameters: {}".format(wf.get_parameters()))
                except KeyError: # if the key 'Pw' is not present in previous_phonondisp_wfs   
                    pass
                
                if not previous_phonondisp_wfs:
                    wf = PhonondispersionWorkflow(params=phonon_params)
                    self.attach_workflow(wf)
                    wf.start()
                    self.append_to_report("Launch PhonondispersionWorkflow (pk= {}) for phonon"
                           " calculation with ecut={} Ry".format(wf.pk, cutoff))
                    self.append_to_report("PhonondispersionWorkflow Parameters: {}".format(wf.get_parameters()))
            
            self.add_attribute('phonon_wf_pks_already_computed',phonon_wf_pks_already_computed)
            
        self.next(self.final_step)
                 
    @Workflow.step    
    def final_step(self):    
        """
        Collect the results in ParameterData with inline calculations
        """  
        
        main_params = self.get_parameters()
        self.append_to_report("Retrieving results of the SsspWorkflow ...")
        
        launch_delta = main_params['input']['compute_delta']
        launch_cohesive = main_params['input']['compute_cohesive']  
        launch_stress = main_params['input']['compute_stress']
        launch_bands = main_params['input']['compute_bands']
        launch_phonon = main_params['input']['compute_phonon'] 
        
        label_dict = main_params['label_dict']
        pseudo_family = main_params['pseudo_family']
        dual = main_params['input']['dual']
        structure = main_params['structure']
        
        # Prepare dictionary with the results to be used for the function 'build_info_inline'
        cutoffs = main_params['input']['cutoffs']  # Cutoffs are in Rydberg
        results_dict = dict( (cutoff, {'parameters_delta':None, 'parameters_energy_bulk':None,
                                       'parameters_phonon_bulk':None,
                                       'parameters_stress':None}) for cutoff in cutoffs )
        
        
        
        if launch_delta:  
                    
            delta_structure = main_params['delta_structure']
            # For compounds I specify which is the main element under investigation 
            # (used in the inline function 'compute_Delta_and_Birch_Murnaghan_EOS_inline' of sssp_utils)
            if len(delta_structure.get_kind_names()) > 1:
                rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
                element = [_ for _ in delta_structure.get_kind_names() if _ in rare_earths][0]
                delta_params_element = ParameterData(dict={'element':element})
            else:
                element = delta_structure.get_kind_names()[0]

            # element = delta_structure.get_kind_names()[0]
            reference_EOS_pk = main_params['delta_parameters']['reference_EOS_pk']
            try:
                reference_EOS = load_node(reference_EOS_pk)
                if not isinstance(reference_EOS, SinglefileData):
                    raise Exception('Reference EOS must be a SingleFileData object')
            except NotExistent:
                raise Exception('Pk of the SingleFileData with reference EOS file does not exist')

            self.append_to_report("Delta factor: obtaining the difference in the EOS w.r.t. {} "
                                 "for element={} and pseudo_family={}".format(reference_EOS.filename, element, label_dict[pseudo_family]))
            
            list_of_wfs = [wf for wf in self.get_step(self.run_cohesive_and_delta).get_sub_workflows() \
                            if (wf.get_attribute('delta_calculation') == True)] \
                + [load_workflow(pk) for pk in self.get_attribute('delta_wf_pks_already_computed')]

            if len(list_of_wfs) != self._default_delta_points: 
                raise Exception('Wrong number of data points for the EOS fit and delta factor computation!')
                    
            try:
                delta_params = dict( (str(wf.get_result('pw_calculation').pk),  wf.get_result('pw_calculation').out.output_parameters) 
                          for wf in list_of_wfs )
                try:
                    delta_params.update({'parameters':delta_params_element})
                except NameError:
                    pass
                delta_results = sssp_utils.get_compute_Delta_and_Birch_Murnaghan_EOS_results(reference_EOS, 
                                                                                    delta_structure, **delta_params)
                for cutoff in results_dict.iterkeys():
                    results_dict[cutoff]['parameters_delta'] = delta_results['output_parameters']
            except ValueError:
                self.append_to_report("Delta factor: WARNING! Some calculations of the EOS for element={} and "
                                      "pseudo_family={} failed."
                                      " I cannot calculate the Delta!".format(element, label_dict[pseudo_family]))
        
        if launch_cohesive:
            
            if len(structure.get_kind_names()) > 1:
                rare_earths_plus_fluorine = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'] + ['F']
                try:
                    element = [_ for _ in structure.get_kind_names() if _ in rare_earths_plus_fluorine][0]
                except IndexError:
                    # Needed in anti ferromagnetic structures 
                    element = ''.join([i for i in structure.get_kind_names()[0] if not i.isdigit()])
            else:
                element = structure.get_kind_names()[0]
                            
            self.append_to_report("Cohesive energy: obtaining the convergence of the cohesive energy w.r.t. the cutoff "
                                 "for element={}, pseudo_family={} and dual={}".format(element, label_dict[pseudo_family],dual))

            list_of_wfs_bulk = [wf for wf in self.get_step(self.run_cohesive_and_delta).get_sub_workflows() \
                            if (wf.get_attribute('cohesive_bulk_calculation') == True)] \
                + [load_workflow(pk) for pk in self.get_attribute('cohesive_bulk_wf_pks_already_computed')]

            list_of_wfs_gas = [wf for wf in self.get_step(self.run_cohesive_and_delta).get_sub_workflows() \
                            if (wf.get_attribute('cohesive_gas_calculation') == True)] \
                + [load_workflow(pk) for pk in self.get_attribute('cohesive_gas_wf_pks_already_computed')]

            list_of_failed_pks = []
            for wf in list_of_wfs_bulk:
                try:
                    pw_calc = wf.get_result('pw_calculation')
                    cutoff = int(round(pw_calc.res.wfc_cutoff / 13.605698066)) # [Ry]
                    results_dict[cutoff]['parameters_energy_bulk'] = pw_calc.out.output_parameters
                except ValueError:
                    list_of_failed_pks.append(wf.pk)
                    
            if list_of_failed_pks:
                self.append_to_report("Cohesive energy: WARNING! Some bulk calculations for element={}, "
                                      "pseudo_family={} and dual={} failed."
                                      " Failed PwWorkflow pks are = {}".format(element, label_dict[pseudo_family],
                                                                        dual, [_ for _ in list_of_failed_pks]))

            list_of_failed_pks = []
            for wf in list_of_wfs_gas:
                try:
                    pw_calc = wf.get_result('pw_calculation')
                    cutoff = int(round(pw_calc.res.wfc_cutoff / 13.605698066)) # [Ry]
                    # In case of alloys (dirty trick...) 
                    for count in xrange(50):
                        try:
                            results_dict[cutoff]['parameters_energy_gas_{}'.format(count)]
                        except KeyError:
                            results_dict[cutoff]['parameters_energy_gas_{}'.format(count)] = pw_calc.out.output_parameters
                            break
                except ValueError:
                    list_of_failed_pks.append(wf.pk)
                    
            if list_of_failed_pks:
                self.append_to_report("Cohesive energy: WARNING! Some gas calculations for element={}, "
                                      "pseudo_family={} and dual={} failed."
                                      " Failed PwWorkflow pks are = {}".format(element, label_dict[pseudo_family],
                                                                        dual, [_ for _ in list_of_failed_pks]))
          
        if launch_stress:
            
            if len(structure.get_kind_names()) > 1:
                rare_earths_plus_fluorine = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'] + ['F']
                try:
                    element = [_ for _ in structure.get_kind_names() if _ in rare_earths_plus_fluorine][0]
                except IndexError:
                    # Needed in anti ferromagnetic structures 
                    element = ''.join([i for i in structure.get_kind_names()[0] if not i.isdigit()])
            else:
                element = structure.get_kind_names()[0]
            
            self.append_to_report("Stress: obtaining the convergence of the cohesive energy w.r.t. the cutoff "
                                 "for element={}, pseudo_family={} and dual={}".format(element, label_dict[pseudo_family],dual))

            list_of_wfs = [wf for wf in self.get_step(self.run_stress).get_sub_workflows() if isinstance(wf,PwWorkflow) ] \
                + [load_workflow(pk) for pk in self.get_attribute('stress_wf_pks_already_computed')]

            list_of_failed_pks = []
            for wf in list_of_wfs:
                try:
                    pw_calc = wf.get_result('pw_calculation')
                    cutoff = int(round(pw_calc.res.wfc_cutoff / 13.605698066)) # [Ry]
                    results_dict[cutoff]['parameters_stress'] = pw_calc.out.output_parameters
                except ValueError:
                    list_of_failed_pks.append(wf.pk)
                    
            if list_of_failed_pks:
                self.append_to_report("Stress: WARNING! Some stress calculations for element={}, "
                                      "pseudo_family={} and dual={} failed."
                                      " Failed PwWorkflow pks are = {}".format(element, label_dict[pseudo_family],
                                                                        dual, [_ for _ in list_of_failed_pks]))
                    
            
        if launch_phonon:
            
            if len(structure.get_kind_names()) > 1:
                rare_earths_plus_fluorine = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'] + ['F']
                try:
                    element = [_ for _ in structure.get_kind_names() if _ in rare_earths_plus_fluorine][0]
                except IndexError:
                    # Needed in anti ferromagnetic structures 
                    element = ''.join([i for i in structure.get_kind_names()[0] if not i.isdigit()])
            else:
                element = structure.get_kind_names()[0]
            
            self.append_to_report("Phonon: obtaining the convergence of the phonon w.r.t. the cutoff "
                                 "for element={}, pseudo_family={} and dual={}".format(element, label_dict[pseudo_family],dual))

            list_of_wfs = [wf for wf in self.get_step(self.run_bands_and_phonon).get_sub_workflows() if \
                           isinstance(wf,PhonondispersionWorkflow) ] \
                           + [load_workflow(pk) for pk in self.get_attribute('phonon_wf_pks_already_computed')]
            
            list_of_failed_pks = []
            for wf in list_of_wfs:
                try:
                    pw_calc = wf.get_parameter('pw_calculation')
                except ValueError:
                    try:
                        pw_calc = wf.get_result('pw_calculation')
                    except ValueError:
                        pass
                try:
                    ph_calc = wf.get_result('ph_calculation')
                    cutoff = int(round(pw_calc.res.wfc_cutoff / 13.605698066)) # [Ry]
                    results_dict[cutoff]['parameters_phonon_bulk'] = ph_calc.out.output_parameters
                except ValueError:
                    list_of_failed_pks.append(wf.pk)
                    
            if list_of_failed_pks:
                self.append_to_report("Phonon: WARNING! Some phonon calculations for element={}, "
                                  "pseudo_family={} and dual={} failed."
                                  " Failed PhonondispersionWorkflow pks are = {}".format(element, label_dict[pseudo_family],
                                                                    dual, [_ for _ in list_of_failed_pks]))

                       
        # Retrieving and saving in a group band structures convergence results      
        if launch_bands:
            try:
                cutoffs = main_params['bands_parameters']['cutoffs']
            except KeyError:
                cutoffs = main_params['input']['cutoffs']
            try:
                noncolin_flag=main_params['bands_pw_parameters']['SYSTEM']['noncolin']
            except KeyError:
                noncolin_flag=False             
            try:
                compound = main_params['compound']
            except KeyError:
                compound = False
            if len(structure.get_kind_names()) > 1:
                if not compound:
                    rare_earths_plus_fluorine = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'] + ['F']
                    element = [_ for _ in structure.get_kind_names() if _ in rare_earths_plus_fluorine][0]
                else:
                    element = structure.get_formula()
            else:
                element = structure.get_kind_names()[0]
            
            self.append_to_report("Bands distance: obtaining the convergence of the bands w.r.t. the cutoff "
                                 "for element={}, pseudo_family={} and dual={}".format(element, label_dict[pseudo_family], dual))
                        
            list_of_wfs = [wf for wf in self.get_step(self.run_bands_and_phonon).get_sub_workflows() if isinstance(wf,PwWorkflow)] \
                        + [load_workflow(pk) for pk in self.get_attribute('band_wf_pks_already_computed')]
            launched_wfs = [wf for wf in self.get_step(self.run_bands_and_phonon).get_sub_workflows() if isinstance(wf,PwWorkflow)]
            
            list_of_pks = []
            list_of_failed_pks = []
            for wf in list_of_wfs:
                try:    
                    list_of_pks.append(wf.get_result('band_structure').pk)
                except ValueError:
                    list_of_failed_pks.append(wf.pk)
            if list_of_failed_pks:
                self.append_to_report("Bands distance: WARNING! Some band calculations for element={}, "
                                      "pseudo_family={} and dual={} failed."
                                      " Failed PwWorkflow pks are = {}".format(element, label_dict[pseudo_family],
                                                                        dual, [_ for _ in list_of_failed_pks]))
               
            bands_group_name = main_params['bands_pw_band_group_name']
            bands_group = Group.get(name=bands_group_name)
          
            # I retrieve the list of cutoffs of the BandsData nodes   
            group_cutoffs = []
            pks_vs_cutoffs = {}
            for pk in list_of_pks:
                bands = load_node(pk)
                cutoff = int(round(bands.inp.output_band.res.wfc_cutoff / 13.605698066))
                pks_vs_cutoffs[pk] = cutoff
                group_cutoffs.append(cutoff)
            
            # Calculate the band distance only if there is the reference cutoff and at least 70% of the expected bands
            if max(cutoffs) != max(group_cutoffs):
                self.append_to_report("Bands distance: the reference cutoff is not present in the group '{}' (pk = {}). "
                                         "I will not calculate the bands distance".format(bands_group_name,bands_group.pk))
            elif float(len(group_cutoffs))/float(len(cutoffs))*100.0 <= 70.0:
                self.append_to_report("Bands distance: the list of cutoffs in the group '{}' (pk = {}) has not enough elements (threshold is 70%). "
                                         "I will not calculate the bands distance".format(bands_group_name,bands_group.pk))
                self.append_to_report("Complete list of cutoffs for the bands distance = {} \n"
                                       "Cutoffs in the group = {}".format(sorted(cutoffs), sorted(group_cutoffs)))
            else:
                max_cutoff = max(group_cutoffs) 
                for pk, cutoff in pks_vs_cutoffs.iteritems():
                    if cutoff == max_cutoff:
                        reference_pk = pk
                        bandsdata_ref = load_node(reference_pk) 
                pks_vs_cutoffs.pop(reference_pk)
                
                if noncolin_flag is True:
                    self.append_to_report('Bands distance: non-collinear calculations found, setting'
                                          ' nspin=1 in bands_distance_inline')
                else:
                    self.append_to_report('Bands distance: standard collinear calculations found, '
                                              'setting nspin=2 in bands_distance_inline')
                    
                results_bands_dict = {}
                warnings_dict = {}
                info_inline_input_dict = {}
                for count,(pk, cutoff) in enumerate(pks_vs_cutoffs.iteritems()):
                    bandsdata1 = load_node(pk)
                    input_dict = {
                                  'smearing': 0.2721,
                                  #'input_nelec': False,
                                  #'nelec': 0.0,
                                  #'nspin': 2,
                                  }
                    if noncolin_flag is True:
                        input_dict['nspin']=1
                    try:
                        is_metal = main_params['bands_parameters']['metal']
                        input_dict['metal'] = is_metal
                    except KeyError:
                        pass    
                   
                    parameters = ParameterData(dict=input_dict)
                    results_bands_distance = sssp_utils.get_bands_distance_results(parameters=parameters,bandsdata1=bandsdata1,
                                                                        bandsdata2=bandsdata_ref)
                    if len(results_bands_distance)>2:
                        raise Exception("Unexcpected behaviour! Found more than a single previous bands_distance_inline calculation")
                    
                    info_inline_input_dict['bands_distance_parameters_{}'.format(count)] = \
                                results_bands_distance['output_parameters']
                
                    warnings_dict['distance_{}_{}_Ry'.format(cutoff, max_cutoff)] = \
                        results_bands_distance['output_parameters'].get_dict()['warnings']
                
                has_warnings = False
                for warning in warnings_dict.itervalues():
                    if len(warning)>0:
                        has_warnings = True    
    
                if has_warnings:    
                    self.append_to_report("Bands distance: ERROR inside the function 'bands_distance_inline'! "
                                          "I will not create a ParameterData for the group '{}' with the "
                                          " bands distances".format(bands_group_name))
                    self.append_to_report("Bands distance: 'bands_distance_inline' warnings are {}".format(warnings_dict))
                else:
                    results_bands_dict, created = sssp_utils.get_bands_distance_info_results(**info_inline_input_dict)
                    bands_distances_info = results_bands_dict['output_parameters']
                    
                    # Reports for debug (maybe to be changed in the future)
                    group_has_parameterdata = False
                    if created:
                        dbnodes = bands_group.nodes.dbnodes
                        for dbnode in dbnodes:
                            node = load_node(dbnode.pk)
                            data_type = type(node)
                            if data_type == ParameterData:
                                if len(launched_wfs) > 0:
                                    self.append_to_report("Bands distance: Group '{}' (pk = {}) already contains old "
                                              "ParameterData (pk = {}). New (created) ParameterData (pk = {}) will be saved in the "
                                              "group while the old one will be deleted".format(bands_group_name,
                                                                        bands_group.pk,node.pk,bands_distances_info.pk))
                                    bands_group.remove_nodes(node)
                                    bands_group.add_nodes(bands_distances_info)
                                else:
                                    self.append_to_report("Bands distance: WARNING: THIS SHOULD NOT HAPPEN! Group '{}' (pk = {}) already contains "
                                              "old ParameterData (pk = {}). New (created) ParameterData (pk = {}) will not be saved "
                                              "in the group".format(bands_group_name,bands_group.pk,node.pk,bands_distances_info.pk))
                                group_has_parameterdata = True 
                        if not group_has_parameterdata:    
                            bands_group.add_nodes(bands_distances_info)
                            self.append_to_report("Bands distance: New (created) ParameterData (pk = {}) with the bands distances "
                                "saved successfully in the group '{}' (pk = {})".format(bands_distances_info.pk, bands_group_name,bands_group.pk) )
                    else:
                        self.append_to_report("Bands distance: ParameterData (pk = {}) with the bands distances was "
                                              "already stored in the database!".format(bands_distances_info.pk))
                        dbnodes = bands_group.nodes.dbnodes
                        for dbnode in dbnodes:
                            node = load_node(dbnode.pk)
                            data_type = type(node)
                            if data_type == ParameterData:
                                if node.pk == bands_distances_info.pk:
                                    self.append_to_report("Bands distance: The ParameterData (pk = {}) with "
                                                          "the bands distances is already "
                                                          "in the group '{}' (pk = {})".format(node.pk,bands_group_name,bands_group.pk))
                                else:
                                    # This should not happen! It adds the new ParameterData in the group and not delete the old one!
                                    self.append_to_report("Bands distance: WARNING: THIS SHOULD NOT HAPPEN! ParameterData (pk = {}) from inline "
                                        "calculation is different from ParameterData (pk = {}) " 
                                        "in the group '{}' (pk = {}). I will add the new one but NOT delete the old one".format(bands_distances_info.pk, node.pk,bands_group_name,bands_group.pk))
                                    bands_group.add_nodes(bands_distances_info)
                                group_has_parameterdata = True       
                        if not group_has_parameterdata:
                            self.append_to_report("Bands distance: WARNING! ParameterData (pk = {}) created in a previous workflow "
                                                  "was not inside the group '{}' (pk = {}). "
                                                  "I will add it".format(bands_distances_info.pk, bands_group_name,bands_group.pk))  
                            bands_group.add_nodes(bands_distances_info)              
 
          
        # Save the info results in a group
        if launch_delta or launch_cohesive or launch_stress or launch_phonon:
            
            group_name = main_params['results_group_name']    
            cutoffs = main_params['input']['cutoffs']  
                
            # Delete the empty entries of the dictionary (TODO: find a better solution!)
            for cutoff in cutoffs:
                l = []
                for k,v in results_dict[cutoff].iteritems():
                    if v is None:
                        l.append(k)
                for k in l:
                    results_dict[cutoff].pop(k)
                       
            group, group_created = Group.get_or_create(name=group_name)
            if group_created:
                self.append_to_report("Final step: Created group '{}' (pk = {}) to store the results "
                    "for element={}, pseudo_family={} and dual={}".format(group_name,group.pk,element,label_dict[pseudo_family],dual))
            else:
                self.append_to_report("Final step: Group '{}' (pk = {}) to store the results was already created "
                    "for element={}, pseudo_family={} and dual={}".format(group_name,group.pk,element,label_dict[pseudo_family],dual))
              
            # List of new workflows launched by the current SsspWorkflow (not bands)
            # It is not used now!
#             launched_wfs = []
#             if launch_delta or launch_cohesive:
#                 launched_wfs += [wf for wf in self.get_step(self.run_cohesive_and_delta).get_sub_workflows()]
#             if launch_stress:
#                 launched_wfs += [wf for wf in self.get_step(self.run_stress).get_sub_workflows()]
#             if launch_phonon:
#                 launched_wfs += [wf for wf in self.get_step(self.run_bands_and_phonon).get_sub_workflows() if \
#                                isinstance(wf,PhonondispersionWorkflow)]   
                
            for cutoff in cutoffs:
                
                # # Check if the stress data are taken from the energy_bulk calculation (it should not be necessary)
                try:
                    if results_dict[cutoff]['parameters_energy_bulk'].pk == results_dict[cutoff]['parameters_stress'].pk:
                        results_dict[cutoff].pop('parameters_stress')
                except KeyError:
                    pass
                
                # Add possibility to give in input parameters_data containing the info_files
                # TODO: maybe to eliminate in the future?     
                try:
                    parameters_data = main_params['input']['parameters_data']
                    if len(cutoffs)>1:
                        raise Exception('The input parameter "parameters_data" can be used only for a single cutoff')
                    for k,v in parameters_data.iteritems():
                        results_dict[cutoff][k] = load_node(v)
                except KeyError:
                    pass    
                
                # Cohesive energy: if there are no bulk parameters I delete all the gas parameters 
                # (otherwise build_info_inline would complain with: 'Too many input parameters')
                try:
                    results_dict[cutoff]['parameters_energy_bulk'] 
                except KeyError:    
                    for count in xrange(50):
                        try:
                            results_dict[cutoff].pop('parameters_energy_gas_{}'.format(count))
                        except KeyError:
                            break
                
                info_parameters, created = sssp_utils.get_build_info_results(**results_dict[cutoff])
                info_params = info_parameters['output_parameters']
                
                self.add_result("Info_file", info_params)
                
                if float(cutoff) != info_params.get_dict()['wfc_cutoff'] and info_params.get_dict()['wfc_cutoff'] != None:
                    raise Exception("Final step: ERROR! Cutoff in ParameterData (pk = {}) is different "
                        "from input cutoff of the SsspWorkflow. There is probably a bug with the "
                        "cutoffs units!".format(info_params.pk))
                
                # Check if the ParameterData was created now or if it was created previously
                if created:
                    self.append_to_report("Final step: ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} "
                        "and dual={} with the info results is "
                        "created now".format(info_params.pk,cutoff,pseudo_family,
                                                                 info_params.get_dict()['dual']))
                else:
                    self.append_to_report("Final step: ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} "
                                "and dual={} with the info results was "
                                "already stored in the database!".format(info_params.pk,cutoff,pseudo_family,
                                                                         info_params.get_dict()['dual']))
                
                # Check if in the group with the results there is already a ParameterData with the same 
                # pseudopotential, dual and cutoff.
                # POLICY: I always replace the old one with the new one.
                group_has_parameterdata = False
                for dbnode in group.nodes.dbnodes:
                    node = load_node(dbnode.pk)
                    if  info_params.get_dict()['wfc_cutoff'] == node.get_dict()['wfc_cutoff'] and \
                        info_params.get_dict()['pseudo_md5'] == node.get_dict()['pseudo_md5'] and \
                        info_params.get_dict()['dual'] == node.get_dict()['dual']:
                        
                        group_has_parameterdata = True
                        if objects_are_equal(info_params.get_dict(),node.get_dict(),epsilon=1e-6):
                            self.append_to_report("Final step: Group '{}' (pk = {}) already contains "
                                    "ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} and dual={} " 
                                    "with the SAME results. 'New' ParameterData (pk = {}) will be saved in the "
                                    "group while the old one will be deleted".format(group_name,group.pk,node.pk,
                                    cutoff,pseudo_family,info_params.get_dict()['dual'],info_params.pk))
                            group.remove_nodes(node)
                            group.add_nodes(info_params)                                 
                        else:
                            self.append_to_report("Final step: Group '{}' (pk = {}) already contains "
                                "ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} and dual={} " 
                                "with DIFFERENT results. WARNING! 'New' ParameterData (pk = {}) will be saved in "
                                "the group while the old one will be deleted".format(group_name,group.pk,node.pk,
                                cutoff,pseudo_family,info_params.get_dict()['dual'],info_params.pk))
                            group.remove_nodes(node)
                            group.add_nodes(info_params)  
                                                           
                if not group_has_parameterdata:    
                    group.add_nodes(info_params)
                    self.append_to_report("Final step: ParameterData (pk = {}) for ecut={} Ry, "
                          "pseudo_family={} and dual={} with info results is added now in the group '{}' "
                          "(pk = {})".format(info_params.pk,cutoff,pseudo_family,
                                             info_params.get_dict()['dual'],group_name,group.pk) )

        self.append_to_report("Ending the SsspWorkflow.")        
        self.next(self.exit)
                        
                    
                    
                

                
##   OLD VERSION                
#                 group_has_parameterdata = False
#                 if created:
#                     dbnodes = group.nodes.dbnodes
#                     for dbnode in dbnodes:
#                         node = load_node(dbnode.pk)                    
#                         if  info_params.get_dict()['wfc_cutoff'] == node.get_dict()['wfc_cutoff'] and \
#                             info_params.get_dict()['pseudo_md5'] == node.get_dict()['pseudo_md5'] and \
#                             info_params.get_dict()['dual'] == node.get_dict()['dual']:
#                             
#                             # pseudo_family_names = UpfData.query(dbattributes__in=models.DbAttribute.objects.filter(key='md5',
#                             #            tval=info_params.get_dict()['pseudo_md5'])).first().get_upf_family_names()
#                             
#                             # TODO: remove this if,else ! Special case if the group is called info_{}
#                             if group_name.startswith('info_'):
#                                 self.append_to_report("Final step: Group '{}' (pk = {}) already contains old "
#                                                 "ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} and dual={}. " 
#                                                 "New (created) ParameterData (pk = {}) will be saved in the group while "
#                                                 "the old one will NOT be deleted".format(group_name,group.pk,node.pk,
#                                                 cutoff,pseudo_family,info_params.get_dict()['dual'],info_params.pk))
#                                 group.add_nodes(info_params)
#                                 group_has_parameterdata = True
#                             else:
#                                 # Replace ParameterData with the results in the group only if new sub-workflows 
#                                 # were launched from the current sssp workflow.                           
#                                 if len(launched_wfs) > 0:
#                                     self.append_to_report("Final step: Group '{}' (pk = {}) already contains old "
#                                                 "ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} and dual={}. " 
#                                                 "New (created) ParameterData (pk = {}) will be saved in the group while "
#                                                 "the old one will be deleted".format(group_name,group.pk,node.pk,
#                                                 cutoff,pseudo_family,info_params.get_dict()['dual'],info_params.pk))
#                                     group.remove_nodes(node)
#                                     group.add_nodes(info_params) 
#                                 else:
#                                     # TODO: maybe we should add the new ParameterData in the group anyway...
#                                     self.append_to_report("Final step: WARNING: THIS SHOULD NOT HAPPEN! Group '{}' (pk = {}) already contains old "
#                                                 "ParameterData (pk = {}) for ecut={} Ry. New (created) ParameterData (pk = {}) will not be saved "
#                                                 "in the group".format(group_name,group.pk,node.pk,cutoff,info_params.pk))
#                                 group_has_parameterdata = True
#                                  
#                     if not group_has_parameterdata:    
#                         group.add_nodes(info_params)
#                         self.append_to_report("Final step: New (created) ParameterData (pk = {}) for ecut={} Ry, "
#                                   "pseudo_family={} and dual={} with info results is added in the group '{}' "
#                                   "(pk = {})".format(info_params.pk,cutoff,pseudo_family,
#                                                      info_params.get_dict()['dual'],group_name,group.pk) )
#                 else:
#                     self.append_to_report("Final step: ParameterData (pk = {}) for ecut={} Ry, pseudo_family={} "
#                                 "and dual={} with the info results was "
#                                 "already stored in the database!".format(info_params.pk,cutoff,pseudo_family,
#                                                                          info_params.get_dict()['dual']))
#                     dbnodes = group.nodes.dbnodes
#                     for dbnode in dbnodes:
#                         node = load_node(dbnode.pk)
#                         if  info_params.get_dict()['wfc_cutoff'] == node.get_dict()['wfc_cutoff'] and \
#                             info_params.get_dict()['pseudo_md5'] == node.get_dict()['pseudo_md5'] and \
#                             info_params.get_dict()['dual'] == node.get_dict()['dual']:
#                             if node.pk == info_params.pk:
#                                 self.append_to_report("Final step: The ParameterData (pk = {}) for ecut={} Ry with "
#                                                       "info results is already "
#                                                       "in the group '{}' (pk = {})".format(node.pk, cutoff, group_name,group.pk))
#                             else:
#                                 self.append_to_report("Final step: WARNING: THIS SHOULD NOT HAPPEN! ParameterData (pk = {}) for ecut={} Ry from inline "
#                                     "calculation is different from ParameterData (pk = {}) " 
#                                     "in the group '{}' (pk = {}). I will add the new one but NOT delete the old one".format(info_params.pk, cutoff, node.pk, group_name, group.pk))
#                                 group.add_nodes(info_params)
#                             group_has_parameterdata = True       
#                     if not group_has_parameterdata:
#                         self.append_to_report("Final step: WARNING! ParameterData (pk = {}) for ecut={} Ry created in a previous workflow "
#                                               "was not inside the group '{}' (pk = {}). "
#                                               "I will add it".format(info_params.pk, cutoff, group_name,group.pk))  
#                         group.add_nodes(info_params)
                                    
