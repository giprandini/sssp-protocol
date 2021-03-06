#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-


import sys
from time import sleep
from copy import deepcopy

'''
=======================================================
Input script for the SsspWorkflow
=======================================================

sys.argv[1] = element
sys.argv[2] = pseudo_family
sys.argv[3] = dual (charge_cutoff = wfc_cutoff*dual)

sys.argv[4] = results_group_name: name of the group for storing the convergence tests results
sys.argv[5] = bands_group_name: name of the group for storing the band structures results

Note: if you want to calculate the bands distance read before the doc of the inline calculation 
      'bands_distance_inline' in sssp_utils.
'''

from sssp_tools.sssp import SsspWorkflow
from aiida.common.example_helpers import test_and_get_code
from aiida.orm import  DataFactory

StructureData = DataFactory('structure')
KpointsData = DataFactory('array.kpoints')
UpfData = DataFactory('upf')

def get_sssp_structures(element):
    """
    Extract structures for SSSP protocol (Cottenier structures here are not reduced to primitive cell!)
    
    # F               ----> Original Cottenier structure for Delta calculation, SiF4 for the other tests
    # Rare-earths     ----> Nitrides for all the tests (primitive cells)
    # Other elements  ----> Original Cottenier structures for all the tests
    """
    rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
    if element == 'F':
        # Cottenier structures for delta calculation (the 71 original elemental structures of Cottenier)
        g = Group.get_from_string('Cottenier_structures')
        nodes = g.nodes.dbnodes
        for node in nodes:
            name = node.attributes['kinds'][0]['name']
            if element == name:
                pk = node.pk
                break
            else:
                pass
        delta_structure = load_node(pk)
        # Cottenier structures reduced to primitive cell + SiF4 + Rare-earths-nitrides(RE-N) (6th row from La to Lu)
        # 86 structures in total (F has two structures: the one from Cottenier and SiF4)
        g = Group.get_from_string('Cottenier_structures_primitive')
        nodes = g.nodes.dbnodes
        for node in nodes:
            if len(node.attributes['kinds'])==2:
                name1 = node.attributes['kinds'][0]['name']
                name2 = node.attributes['kinds'][1]['name']
                if sorted([name1,name2]) == ['F','Si']:
                    pk = node.pk
                    break
            else:
                pass
        structure = load_node(pk)
    elif element in rare_earths:
        g = Group.get_from_string('Cottenier_structures_primitive')
        nodes = g.nodes.dbnodes
        for node in nodes:
            if len(node.attributes['kinds'])==2: # Nitrides
                name1 = node.attributes['kinds'][0]['name']
                name2 = node.attributes['kinds'][1]['name']
                if sorted([name1,name2]) == sorted(['N','{}'.format(element)]):
                    pk = node.pk
                    break
            else:
                pass
        delta_structure = load_node(pk)
        structure = load_node(pk)
    else:    
        # Cottenier structures for delta calculation (the 71 original elemental structures of Cottenier)
        g = Group.get_from_string('Cottenier_structures')
        nodes = g.nodes.dbnodes
        for node in nodes:
            name = node.attributes['kinds'][0]['name']
            if element == name:
                pk = node.pk
                break
            else:
                pass
        delta_structure = load_node(pk)
        structure = load_node(pk)

    return delta_structure, structure



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


# Dictionary with pseudopotential family names in AiiDA and their corresponding 'official' short names.
pseudo_dict = {                    
              'pslib.0.3.1_PBE_US': '031US',
              'pslib.0.3.1_PBE_PAW': '031PAW',
              'pslib.1.0.0_PBE_US': '100US',
              'pslib.1.0.0_PBE_PAW': '100PAW',
              'pslib.1.0.0_PBE_US_low_acc': '100US_low',
              'pslib.1.0.0_PBE_PAW_low_acc': '100PAW_low',
#              'pslib.orig_PBE_PAW': 'psorigPAW',
#              'pslib.orig_PBE_US': 'psorigUS',
              'GBRV_1.2': 'GBRV-1.2',
              'GBRV_1.4': 'GBRV-1.4',
              'GBRV_1.5': 'GBRV-1.5',
              'SG15': 'SG15',
              'SG15_1.1': 'SG15-1.1',
              'THEOS': 'THEOS',
#              'THEOS_alternative': 'THEOS2',
              #'RE_100_PAW': '100PAW',
              #'RE_100_US': '100US',
              'RE_Wentz': 'Wentzcovitch',
              'RE_Wentz_plus_nitrogen': 'Wentzcovitch',
              'RE_pslib.1.0.0_PBE_US_plus_nitrogen': '100US',
              'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen': '100PAW',
              'RE_Vander_plus_nitrogen': 'RE_Vanderbilt',
              'RE_GBRV_plus_nitrogen': 'RE_GBRV',
              'RE_SG15_plus_nitrogen': 'RE_SG15',
              'Goedecker': 'Goedecker',
              'Dojo': 'Dojo',
              'Dojo_v0.4': 'Dojo-0.4',
#              'All_Electrons': 'all_elec',
#              'All_Electrons_denser': 'all_elec_denser',
#              'PBE_US_BM': 'BM',
#              'PBE_US_GIPAW': 'GIPAW',
               'atomic_NC_test':'Mn_NC_atomic',
               'Vanderbilt_US_test': 'Mn_Fe_van',
               'GBRV-1.4_LDA': 'GBRV-1.4_LDA',
               'GBRV-1.4_PBEsol': 'GBRV-1.4_PBEsol',
               'GBRV-1.5_LDA': 'GBRV-1.5_LDA',
               'GBRV-1.5_PBEsol': 'GBRV-1.5_PBEsol',
                                  }

# Specify which SSSP tests to launch
send = True
compute_delta = True
compute_bands = False
compute_cohesive = False
compute_stress = True
compute_phonon = True

try:
    element = sys.argv[1]
except IndexError:
    print >> sys.stderr, ("The first parameter must be the element")
    sys.exit(1)

delta_structure, structure = get_sssp_structures(element) 

valid_pseudo_groups = UpfData.get_upf_groups(filter_elements=element)
try:
    pseudo_family = sys.argv[2]
except IndexError:
    print >> sys.stderr, "The second parameter must be the pseudo family name"
    print >> sys.stderr, "Valid UPF families are:"
    print >> sys.stderr, "\n".join("* {}".format(i.name) for i in valid_pseudo_groups)
    sys.exit(1)
try:
    UpfData.get_upf_group(pseudo_family)
except NotExistent:
    print >> sys.stderr, "pseudo_family='{}',".format(pseudo_family)
    print >> sys.stderr, "but no group with such a name found in the DB."
    print >> sys.stderr, "Valid UPF groups are:"
    print >> sys.stderr, ",".join(i.name for i in valid_pseudo_groups)
    sys.exit(1)

try:
    dual = int(sys.argv[3])
except IndexError:
    print >> sys.stderr, ("The third parameter must be the dual")
    sys.exit(1)

# # User-specified group_name where to store the info results and bands results.
# # If not specified the default name is 'element_pseudo_dual_sssp' 
try:
    results_group_name = sys.argv[4]
except IndexError:
    results_group_name = '{}_{}_{}_sssp'.format(element, pseudo_dict[pseudo_family], dual)
    print >> sys.stdout, ("Name of group for storing info data results not specified. I will use the default "
                          "name: {}".format(results_group_name))   
try:
    bands_group_name = sys.argv[5]
except IndexError:
    bands_group_name = '{}_{}_{}_sssp_bands'.format(element, pseudo_dict[pseudo_family], dual)
    print >> sys.stdout, ("Name of group for storing band structures/distances data not specified. I will use the default "
                          "name: {}".format(bands_group_name))
if results_group_name == bands_group_name:
    print >> sys.stderr, ("Error: Groups with info data results and band structures/distances data must have different names")
    sys.exit(1)

rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']

# List of cutoffs for the convergence tests (in Rydberg)
cutoffs = range(30,85,5)+[90,100,120,150,200]
cutoffs = [40,60,80,200]   # !!! DEBUG
bands_cutoffs = deepcopy(cutoffs)

# codenames
pw_codename = 'pw6.0@fidis'
ph_codename = 'ph6.0@fidis'

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

wien2k_EOS_pk = 207189 # Pk of SingleFileData with the 'Wien2k.txt' file from calcDelta package (version 3.1)
if element in rare_earths:
    wien2k_EOS_pk = 780061 # Pk of SingleFileData with the 'Wien2k_RE_nitrides.txt' file for the rare-earth nitrides (courtesy of M. Topsakal) 
if pseudo_family in ['SG15', 'SG15_1.1', 'Goedecker','atomic_NC_test']:
    delta_dual = 4
else:
    delta_dual = 8

# cohesive parameters
cohesive_gas_degauss = 0.1  
cohesive_bulk_degauss = 0.02
if element == 'O' or element in rare_earths:
    cohesive_bulk_kpoints_mesh = [10,10,10]
else:
    cohesive_bulk_kpoints_mesh = [6,6,6]
cohesive_gas_structures_group_name = 'Isolated_atoms'

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
bands_kpoints = load_node(67853) # 6x6x6 with no-offset 

# phonon parameters 
phonon_degauss = 0.02  # Rydberg
qpoint_crystal = [0.5, 0.5, 0.5]
if element == 'O' or element in rare_earths:
    phonon_kpoints_mesh = [10,10,10]
else:
    phonon_kpoints_mesh = [6,6,6]
settings = {'cmdline':['-nk',str(4)]}

# validate
test_and_get_code(pw_codename,'quantumespresso.pw')
test_and_get_code(ph_codename,'quantumespresso.ph')
validate_upf_family(pseudo_family, structure.get_kind_names())

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
                                      'reference_EOS_pk': wien2k_EOS_pk,
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
