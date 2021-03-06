#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-


import sys
from time import sleep
from copy import deepcopy

'''
=======================================================
Test example for input script of the SsspWorkflow
=======================================================

sys.argv[1] = pk of the StructureData of the Pd crystal structure
sys.argv[2] = pw codename
sys.argv[3] = ph codename
sys.argv[4] = pk of the SinglefileData of the 'Wien2k.txt' file from calcDelta package (version 3.1)
sys.argv[5] = pk of the KpointsData used for the calculation of the band distance for band structure convergence

'''

from sssp_tools.sssp import SsspWorkflow
from aiida.common.example_helpers import test_and_get_code
from aiida.orm import  DataFactory
from sssp_tools.sssp_utils import pseudo_families_libraries_dict  as pseudo_dict

StructureData = DataFactory('structure')
KpointsData = DataFactory('array.kpoints')
UpfData = DataFactory('upf')


def validate_upf_family(pseudo_family, all_species):
    """
    Validate if the pseudo_family is correct.
    """
    elements = all_species
    UpfData = DataFactory('upf')
    valid_families = UpfData.get_upf_groups(filter_elements=elements)
    valid_families_names = [family.name for family in valid_families]
    if pseudo_family not in valid_families_names:
        raise ValueError("Invalid pseudo family '{}'. "
                         "Valid family names are: {}".format(
            pseudo_family, ",".join(valid_families_names)))

def get_Zvalence_from_pseudo(pseudo):
     """
     Extract the number of valence electrons from a pseudo
     """
     with open(pseudo.get_file_abs_path(),'r') as f:
         lines=f.readlines()
         for line in lines:
             if 'valence' in line:
                 try:
                     return int(float(line.split("z_valence=\""
                                             )[-1].split("\"")[0].strip()))
                 except (ValueError, IndexError):
                     try:
                         return int(float(line.split("Z")[0].strip()))
                     except (ValueError, IndexError):
                         return None

# Specify which SSSP tests to launch
send = True
compute_delta = True
compute_bands = True
compute_cohesive = True
compute_stress = True
compute_phonon = True

pseudo_family = 'pslib.0.3.1_PBE_PAW'
dual= 8
element = 'Pd'
cohesive_gas_structures_group_name = 'Isolated_atoms'
# List of wavefunction cutoffs used in the example for the SSSP convergence tests (in Rydberg)
cutoffs = [30,40,50,200]   
bands_cutoffs = deepcopy(cutoffs)
results_group_name = '{}_{}_{}_sssp'.format(element, pseudo_dict[pseudo_family], dual)
bands_group_name = '{}_{}_{}_sssp_bands'.format(element, pseudo_dict[pseudo_family], dual)

try:
    structure_pk = sys.argv[1]
except IndexError:
    print >> sys.stderr, ("The first parameter must be the structure pk")
    sys.exit(1)

delta_structure = load_node(int(structure_pk))
structure = load_node(int(structure_pk))

try:
    pw_codename = sys.argv[2]
except IndexError:
    print >> sys.stderr, "The second parameter must be the pw code name"
    sys.exit(1)

try:
    ph_codename = sys.argv[3]
except IndexError:
    print >> sys.stderr, "The third parameter must be the ph code name"
    sys.exit(1)

# validate codes and pseudopotential family
test_and_get_code(pw_codename,'quantumespresso.pw')
test_and_get_code(ph_codename,'quantumespresso.ph')
validate_upf_family(pseudo_family, structure.get_kind_names())

try:
    wien2k_EOS_pk = sys.argv[4]
except IndexError:
    print >> sys.stderr, ("The fourth parameter must be the pk of the Wien2k SinglefileData")
    sys.exit(1)

try:
    band_kpoints_pk = sys.argv[5]
except IndexError:
    print >> sys.stderr, ("The fifth parameter must be the pk of the KpointsData used for band structure convergence")
    sys.exit(1)

valid_pseudo_groups = UpfData.get_upf_groups(filter_elements=element)
try:
    UpfData.get_upf_group(pseudo_family)
except NotExistent:
    print >> sys.stderr, "pseudo_family='{}',".format(pseudo_family)
    print >> sys.stderr, "not found in the DB. Please create pseudo_family with this name."
    sys.exit(1)

rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']

# input
target_time_seconds = 60*60*1.0
max_time_seconds = target_time_seconds*2.0
max_num_machines = 2
#calculation_set_dict={'custom_scheduler_commands': '#SBATCH --constraint=gpu \n#SBATCH --account=ch3'
#          ' \n#SBATCH --partition=normal \n#SBATCH --gres=gpu:0'}
calculation_set_dict={}

## delta parameters 
delta_wfc_cutoff = 200.0
delta_kpoints_mesh = [20,20,20]    
delta_degauss = 0.002 

conv_thr = 1.e-10
mixing_beta = 0.3
finish_with_scf = False

# cohesive parameters
cohesive_gas_degauss = 0.1  
cohesive_bulk_degauss = 0.02
if element == 'O' or element in rare_earths:
    cohesive_bulk_kpoints_mesh = [10,10,10]
else:
    cohesive_bulk_kpoints_mesh = [6,6,6]

# stress parameters
if element == 'O' or element in rare_earths:
    stress_kpoints_mesh = [10,10,10]
else:
    stress_kpoints_mesh = [6,6,6]
stress_degauss = 0.02

## bands parameters
bands_degauss = 0.02
bands_scf_kpoints_mesh = [20,20,20] 
number_of_bands_nelec_units = 2.0  # number_of_bands = (number_of_electrons)*(number_of_bands_nelec_units)
bands_kpoints = = load_node(int(band_kpoints_pk)) # 6x6x6 with no-offset 

# phonon parameters 
phonon_degauss = 0.02  # Rydberg
qpoint_crystal = [0.5, 0.5, 0.5]
if element == 'O' or element in rare_earths:
    phonon_kpoints_mesh = [10,10,10]
else:
    phonon_kpoints_mesh = [6,6,6]
## setting npool equal to 4
# settings = {'cmdline':['-nk',str(4)]}


# Find Z valence for the rare-earth pseudopotentials
if element in rare_earths:
    pseudo_family_group = UpfData.get_upf_group(pseudo_family)
    for pseudo in pseudo_family_group.nodes:
        if element == pseudo.element:
            pseudo_RE = pseudo
            break
        else:
            pass
    z_valence_RE = get_Zvalence_from_pseudo(pseudo_RE)
else:
    z_valence_RE = 0


# Workflow parameters
wf_parameters = {
                 # General inputs of the sssp workflow   
                 'pw_codename': pw_codename,
                 'ph_codename': ph_codename,
                 'pseudo_family': pseudo_family,
                 'structure': structure,
                 'label_dict': pseudo_dict,
                 'results_group_name': results_group_name,
                 
                 'input': {
                           'compute_delta': compute_delta,
                           'compute_cohesive': compute_cohesive,
                           'compute_stress': compute_stress,
                           'compute_bands': compute_bands,
                           'compute_phonon': compute_phonon,
                           
                           'cutoffs': cutoffs, # list of cutoffs for the sssp convergence tests
                           'dual': dual,       # dual for the convergence tests
                           
                           # 'parameters_data': parameters_data (for build_info_inline),
                           },
                 
                 # Inputs specific to the delta computation
                 'delta_structure': delta_structure,
                 'delta_parameters': {
                                      'reference_EOS_pk': int(wien2k_EOS_pk),
                                      'kpoints_mesh': delta_kpoints_mesh,           
                                      },
                 'delta_pw_parameters': {
                                         'SYSTEM' :{
                                                    'ecutwfc': delta_wfc_cutoff,
                                                    'ecutrho': delta_wfc_cutoff*delta_dual,
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': delta_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       'mixing_beta': mixing_beta,
                                                       },
                                         },
                 'delta_pw_input': {
                             'finish_with_scf': finish_with_scf,
                             'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines
                                                     },
                                    },
                 #'delta_pw_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'delta_pw_settings': {},
                 'delta_pw_calculation_set': calculation_set_dict,
                 
                 # Inputs specific to the cohesive energy computation (Bulk)
                 'cohesive_pw_bulk_parameters': {
                                        'CONTROL':{
                                                   'tstress': True,
                                                   },
                                         'SYSTEM' :{
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': cohesive_bulk_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       },
                                         },
                 'cohesive_bulk_parameters': {
                                      'kpoints_mesh': cohesive_bulk_kpoints_mesh,           
                                      },
                 'cohesive_pw_bulk_input': {
                             'finish_with_scf': finish_with_scf,
                             'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                            },
                 #'cohesive_pw_bulk_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'cohesive_pw_bulk_settings': {},
                 'cohesive_pw_bulk_calculation_set': calculation_set_dict,
                 
                 # Inputs specific to the cohesive energy computation (Gas)
                 #'cohesive_gas_structure': gas_structure,
                 'cohesive_gas_structures_group_name': cohesive_gas_structures_group_name,
                 'cohesive_pw_gas_parameters': {
                                         'SYSTEM' :{
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': cohesive_gas_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       },
                                         },
                 'cohesive_pw_gas_input': {
                             'finish_with_scf': finish_with_scf,
                             'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                            },
                 #'cohesive_pw_gas_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'cohesive_pw_gas_settings': {},
                 'cohesive_pw_gas_calculation_set': calculation_set_dict,
                 
                 # Inputs specific to the stress computation
                 'stress_pw_parameters': {
                                         'CONTROL':{
                                                    'tstress': True,
                                                    },
                                         'SYSTEM' :{
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': stress_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       },
                                         },
                 'stress_parameters': {
                                      'kpoints_mesh': stress_kpoints_mesh,           
                                      },
                 'stress_pw_input': {
                             'finish_with_scf': finish_with_scf,
                             'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                            },
                 #'stress_pw_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'stress_pw_settings': {},
                 'stress_pw_calculation_set': calculation_set_dict,
                 
                 # Inputs specific to the band computation
                 'bands_pw_parameters': {
                                         'CONTROL':{
                                                    },
                                         'SYSTEM' :{
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': bands_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       },
                                         },
                 'bands_parameters': {
                                      'cutoffs': bands_cutoffs,
                                      'kpoints_mesh': bands_scf_kpoints_mesh,           
                                      },
                 'bands_pw_input': {
                             'finish_with_scf': finish_with_scf,
                             'clean_workdir': True,
                             'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                            },
                 'bands_pw_band_kpoints': bands_kpoints,
                 'bands_pw_band_input': {
                                         'clean_workdir': True,
                                         'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                         #'number_of_bands': bands_number_of_bands,
                                         'number_of_bands_nelec_units': number_of_bands_nelec_units,
                                         },
                 'bands_pw_band_parameters_update': {
                                            'SYSTEM':{
                                                      'noinv':True,
                                                      'nosym':True,
                                                      },         
                                            'ELECTRONS':{
                                                         'diagonalization':'cg',
                                                         },
                                                     },
                 'bands_pw_band_group_name': bands_group_name,
                 #'bands_pw_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'bands_pw_settings': {},
                 'bands_pw_calculation_set': calculation_set_dict,
                 #'bands_pw_band_settings': {}, 
                 'bands_pw_band_calculation_set': calculation_set_dict,
                 
                 # Inputs specific to the phonon computation
                 'phonon_parameters': {
                                      'kpoints_mesh': phonon_kpoints_mesh,
                                      'qpoint': qpoint_crystal,           
                                      },
                 'phonon_pw_parameters': {
                                         'CONTROL':{
                                                    'wf_collect': True,
                                                    },
                                         'SYSTEM' :{
                                                    'occupations': 'smearing',
                                                    'smearing': 'mv',
                                                    'degauss': phonon_degauss,
                                                    },
                                         'ELECTRONS': {
                                                       'conv_thr': conv_thr,
                                                       },
                                         },
                 'phonon_pw_input': {
                                     'finish_with_scf': finish_with_scf,
                                     'automatic_parallelization':
                                                    {
                                                     'max_wall_time_seconds': max_time_seconds,
                                                     'target_time_seconds': target_time_seconds,
                                                     'max_num_machines': max_num_machines,
                                                     },
                                    },
                 
                 'phonon_ph_parameters': {
                                         'INPUTPH':{
                                                    'tr2_ph' : 1e-16,
                                                    },
                                         },
                 'phonon_ph_input': {
                                     'use_qgrid_parallelization': False,
                                     'clean_workdir': True,
                                    },
                 # Is it possible to do an automatic parallelization on the phonon?
                 'phonon_ph_calculation_set': {
                                               'resources':{'num_machines': max_num_machines},
                                               'max_wallclock_seconds':max_time_seconds,
                                               #'custom_scheduler_commands': calculation_set_dict['custom_scheduler_commands'],
                                               },
                 
                 #'phonon_pw_kpoints' : user-defined KpointsData() node. It has the priority w.r.t. the kpoints_mesh. 
                 #'phonon_ph_qpoints': {}, user-defined KpointsData() node. It has the priority w.r.t. the qpoint
                 'phonon_pw_calculation_set': calculation_set_dict,
                 #'phonon_pw_settings': {},
                 'phonon_ph_settings': settings,             
                 }


wf = SsspWorkflow(params = wf_parameters)

if send:
    wf.start()
    print ("Launch SsspWorkflow {}".format(wf.pk))
    print ("Parameters: {}".format(wf.get_parameters()))
else:
    print ("Would launch SsspWorkflow")
    print ("Parameters: {}".format(wf.get_parameters()))

# Write SsspWorkflow pk in a file (to be commented if not wanted).
if send:
    with open(element+'_'+pseudo_dict[pseudo_family]+'_'+str(dual)+'_pk.ssspworkflow','a') as o:
        o.write(str(wf.pk))
        o.write('\n')
    sleep(5)
