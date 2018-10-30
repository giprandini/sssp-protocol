
from aiida.orm.user import User
from __builtin__ import True
from aiida.workflows.user.epfl_theos.quantumespresso.sssp import SsspWorkflow
from aiida.workflows.user.epfl_theos.quantumespresso import sssp_utils
from aiida.workflows.user.epfl_theos.dbimporters.utils import objects_are_equal, objects_set
from aiida.common.example_helpers import test_and_get_code
import sys, subprocess, os, json
from time import sleep
from matplotlib.compat.subprocess import CalledProcessError
from aiida.workflows.user.epfl_theos.quantumespresso.sssp_utils import pseudo_families_libraries_dict  as pseudo_dict
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import math, matplotlib
from calcDelta import calcDelta

ParameterData = DataFactory('parameter')
UpfData = DataFactory('upf')
BandsData = DataFactory('array.bands')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]



'''
Script to generate the png files with the SSSP convergence pattern plots for each element.

It needs:

- json files with SSSP libraries and cutoffs

- AiiDA groups with the info files produced by the SsspWorkflow

- SingleDataFiles with the all-electron data for the delta-factor test

'''

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, prog='sssp_convergence_plot',
    description="""
This AiiDA script generates the png files cntaining the SSSP convergence pattern plots for each element.

Example:
$ verdi run sssp_convergence_plot.py eff   --> it generates the convergence plots for the SSSP efficiency library
""")

parser.add_argument("sssp_lib", type=str, help="SSSP library: 'efficiency' or 'precision'")
args = parser.parse_args()


#### 
# Pk of SingleFileData with all-electron data for delta-factor test
reference_EOS_pk = 207189           # Cottenier elemental crystals
reference_EOS_REN_pk = 780061       # RE nitrides
# Files with SSSP tables (without $HOME in the path)
file_SSSP_efficiency = '/Dropbox/sssp/sssp_v1.1/efficiency/sssp_efficiency.json'
file_SSSP_preciswion = '/Dropbox/sssp/sssp_v1.1/precision/sssp_precision.json'
####


# Take all groups with the infos: 'info_ELEMENT_sssp1'
groups=Group.query(name__startswith='info_',name__endswith='_sssp1')
#groups=Group.query(name__startswith='La_')

# Take all the ParameterData containing the infos
infos = ParameterData.query(dbgroups__pk__in=[_.pk for _ in groups])

## Take all the elements in the groups 
# elements = set(infos.filter(dbattributes__key='element').values_list('dbattributes__tval',flat=True))

# All 85 elements tested in the SSSP
elements = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',  
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
            'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te', 
            'I','Xe','Cs','Ba','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','Rn',
            'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
#elements = ['La']   # DEBUG

# Reference all-electron data for delta-factor test
reference_EOS = load_node(reference_EOS_pk)          # SingleFileData with the 'Wien2k.txt' file from calcDelta package (version 3.1)
reference_EOS_REN = load_node(reference_EOS_REN_pk)      # SingleFileData with the 'Wien2k_RE_nitrides.txt' file for the rare-earth nitrides (courtesy of M. Topsakal)

sssp_type = args.sssp_lib
if sssp_type in ['eff', 'efficiency']:
    sssp_efficiency = True
elif sssp_type in ['prec','precision']:
    sssp_efficiency = False
else:
    raise Exception("You need to specify 'precision' or 'efficiency' for the convergence pattern plot")

# Open json files (downloadable from http:/materialscloud/sssp/) with SSSP libraries and cutoffs
with open(os.path.expanduser('~{}'.format(file_SSSP_efficiency)),'r') as o:
    sssp_eff_table = json.load(o)
with open(os.path.expanduser('~{}'.format(file_SSSP_preciswion)),'r') as o:
    sssp_prec_table = json.load(o)

# List of cutoffs used in the convergence pattern plots: 16 cutoffs from 30 Ry to 200 Ry.
list_of_cutoffs = range(30,85,5)+[90,100,120,150,200]

rytoev = 13.605698066
pseudos_colors_dict = dict([(pseudo,color) for pseudo,color in zip(
             ['100PAW','100PAW_low','100US','100US_low','031PAW','031US',
              'GBRV-1.2','GBRV-1.4','GBRV-1.5','SG15','SG15-1.1','Goedecker',
              'Dojo','THEOS','Wentzcovitch','Vanderbilt','THEOS2','all_elec',
              'all_elec_denser','BM','GIPAW','psorigPAW','psorigUS'],
             ['#008B00','#80FF80','#FF0000','#FF8080','#FF00FF','#0000FF',
              '#00CDCD','#4682B4','#B8860B','#000000','#708090','#808000',
              '#FFA500','#D7DF01','#610B5E','#8FBC8F','#F0F000','#F000F0',
              '#00F0F0','#A5FF00','#B44682','#CD00CD','#86B80B']
              )])
pseudos_order = [
                'RE_Wentz_plus_nitrogen',
                'RE_pslib.1.0.0_PBE_US_plus_nitrogen',
                'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen', 
                 'RE_Wentz','THEOS','SG15','SG15_1.1','Goedecker',
                'Dojo','GBRV_1.2','GBRV_1.4','GBRV_1.5',
                'pslib.0.3.1_PBE_US','pslib.0.3.1_PBE_PAW','pslib.1.0.0_PBE_US',
                'pslib.1.0.0_PBE_PAW','pslib.1.0.0_PBE_US_low_acc','pslib.1.0.0_PBE_PAW_low_acc',
                'RE_Vander','THEOS_alternative','All_Electrons','All_Electrons_denser',
                'PBE_US_BM','PBE_US_GIPAW','pslib.orig_PBE_US','pslib.orig_PBE_PAW',
                'RE_Wentz_plus_nitrogen','RE_pslib.1.0.0_PBE_US_plus_nitrogen',
                'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen',
                ]

# Norm-conserving pseudopotential libraries: used to choose always dual=4
norm_conserving_pseudo_families = ['SG15','SG15_1.1','Goedecker','Dojo']

# Elements with max phonon frequency < 100 cm^{-1}
elements_low_freq_ph = ['Ar','Ba','Cs','Fe','He','Hg','In','K','Kr','Pb','Po','Rb','Rn','Sr','Tl','Xe']
# Lanthanides
rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']


# Loop on all elements specified 
for element in elements:
    print '*****************'
    print 'Element = {}'.format(element)
    print '*****************'
    
    # AiiDA pseudo families
    pseudo_families = [
                    'pslib.0.3.1_PBE_US',
                    'pslib.0.3.1_PBE_PAW',
                    'pslib.1.0.0_PBE_US',
                    'pslib.1.0.0_PBE_PAW',
                    'GBRV_1.2',
                    'GBRV_1.4',
                    'GBRV_1.5',
                    'SG15',
                    'SG15_1.1',
                    'THEOS',
                    'Goedecker',
                    'Dojo'

                   ]
    
    if element in rare_earths:
        pseudo_families = [
                            'RE_Wentz_plus_nitrogen',
                            'RE_pslib.1.0.0_PBE_US_plus_nitrogen',
                            'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen',
                            #'RE_GBRV_plus_nitrogen',
                           ]
    
    # Band groups with the bands distances
    groups_bands=Group.query(name__startswith='{}_'.format(element),name__endswith='_bands')
    infos_bands = ParameterData.query(dbgroups__pk__in=[_.pk for _ in groups_bands])
           
    # Retrieve all info files for a given element
    infos_for_element = infos.filter(dbattributes__key='element',dbattributes__tval=element)
    
    ## Take all pseudos launched for a given element
    pseudos_md5 = set(infos.filter(dbattributes__key='pseudo_md5').filter(dbattributes__key='element',
                    dbattributes__tval=element).values_list('dbattributes__tval',flat=True))

    # Find pseudos for which the full SSSP protocol was not launched (for a given element)
    pseudos_to_launch = []
    for pseudo_family in pseudo_families:
        pseudo_family_group = UpfData.get_upf_group(pseudo_family)
        for pseudo in pseudo_family_group.nodes:
            if element == pseudo.element:
                if pseudo.md5sum not in pseudos_md5:
                    pseudos_to_launch.append(pseudo_family)
                    print ''
                    print '!!! WARNING !!! Missing pseudopotential library: {}'.format(pseudo_family)
                    print ''
    
    # Get all duals for a given element                
    duals = set([(p.get_dict()['dual']) for p in infos_for_element if p.get_dict()['dual'] != None ])
    # Loop on the list of duals
    for dual in duals-{4.0}:
        
        print ''
        print '---> Dual = {}'.format(dual)
        print ''
        
        # Retrieve all info files for a given element and dual  
        infos_for_dual = [p for p in infos_for_element if p.get_dict()['dual'] == dual]    

        offset = 8
        count = 0
        max_cutoff = 0
        deltas = []
        delta1s = []
        Zs = []
        max_freqs = []
        labels = []
        pressure_unit = []
        delta_pressure_str = []
        plot_lines = []

        cutoffs = set([p.get_dict()['wfc_cutoff'] for p in infos_for_dual])
        
        plt.figure(figsize=(40,18))

        qpoints = objects_set([p.get_dict()['q-point'] for p in infos_for_dual
                               if p.get_dict()['q-point'] != None],epsilon=1e-4)
        if len(qpoints)!=1:
            raise ValueError("There should be at least one and only one q-point computed!")
    
        pressure_units = list(set([p.get_dict()['pressure_units'] for p in infos_for_dual]))
        if len(pressure_units)>1:
            raise ValueError("There should be only one kind of pressure units!")
        
        phonon_units = list(set([p.get_dict()['phonon_frequencies_units'] for p in infos_for_dual]))
        if len(phonon_units)>1:
            raise ValueError("There should be only one kind of phonon frequency units!")
        
        energy_units = list(set([p.get_dict()['cohesive_energy_units'] for p in infos_for_dual]))
        if len(energy_units)>1:
            raise ValueError("There should be only one kind of cohesive energy units!")

        energy_unit = energy_units[0].replace('atom','at')
        phonon_unit = phonon_units[0]
        qpoint = qpoints[0]        


        # Read Wien2k file (for alternative method to calculate the pressure)
        if element in rare_earths:
            wien2k_data = np.loadtxt(reference_EOS_REN.get_file_abs_path(), 
                dtype={'names': ('element', 'V0', 'B0', 'B1'),
                'formats': ('S2', np.float, np.float, np.float)})
            wien2k_dict = dict([(e,{'B0':B0,'B1':B1,'V0':V0}) for e,V0,B0,B1 in wien2k_data])
            B0 = wien2k_dict[element]['B0']
            B1 = wien2k_dict[element]['B1']
            V0 = wien2k_dict[element]['V0']
        elif element == 'F':
            B0 = None
        else:
            wien2k_data = np.loadtxt(reference_EOS.get_file_abs_path(), 
                dtype={'names': ('element', 'V0', 'B0', 'B1'),
                'formats': ('S2', np.float, np.float, np.float)})
            wien2k_dict = dict([(e,{'B0':B0,'B1':B1,'V0':V0}) for e,V0,B0,B1 in wien2k_data])
            B0 = wien2k_dict[element]['B0']
            B1 = wien2k_dict[element]['B1']
            V0 = wien2k_dict[element]['V0']      
    
        # Loop on the pseudos present in the groups (ordered for the SSSP plot)    
        for pseudo_family in sorted(list(set(pseudo_families) - set(pseudos_to_launch)),
                                    key=lambda x: -pseudos_order.index(x)):
    
            pseudo_family_has_element = False
            pseudo_family_group = UpfData.get_upf_group(pseudo_family)
            for pseudo in pseudo_family_group.nodes:
                if element == pseudo.element:
                    pseudo_md5sum = pseudo.md5sum
                    pseudo_family_has_element = True
        
            if pseudo_family_has_element:    
    
                print ''
                print '------> Pseudopotential library = {}'.format(pseudo_family)
                print ''
                
                ### BAND PART
                has_bands = True
                infos_bands_for_pseudo_and_dual = []
                for info in infos_bands:
                    for i in xrange(16):
                        try:
                            wfc_cutoff_tmp = eval('info.inp.output_parameters.inp.bands_distance_parameters_{}.inp.output_parameters.inp.bandsdata1.inp.output_band.res.wfc_cutoff'.format(i))
                            rho_cutoff_tmp = eval('info.inp.output_parameters.inp.bands_distance_parameters_{}.inp.output_parameters.inp.bandsdata1.inp.output_band.res.rho_cutoff'.format(i))
                            dual_tmp = round(rho_cutoff_tmp/wfc_cutoff_tmp,1)
                            pseudo_md5sum_tmp = eval('info.inp.output_parameters.inp.bands_distance_parameters_{}.inp.output_parameters.inp.bandsdata1.inp.output_band.inp.pseudo_{}.md5sum'.format(i,element))
                            if pseudo_family in norm_conserving_pseudo_families:
                                if dual_tmp == 4.0 and pseudo_md5sum_tmp == pseudo_md5sum:
                                    infos_bands_for_pseudo_and_dual.append(info)    
                                if dual_tmp == 8.0 and pseudo_md5sum_tmp == pseudo_md5sum:
                                    infos_bands_for_pseudo_and_dual.append(info)
                                    print("     !!! WARNING !!! This norm-conserving pseudopotential has dual=8.0 ")                                            
                            else:    
                                if dual_tmp == dual and pseudo_md5sum_tmp == pseudo_md5sum:
                                    infos_bands_for_pseudo_and_dual.append(info)
                            break
                        except AttributeError:
                            continue
                
                if len(infos_bands_for_pseudo_and_dual)>1:
                    raise ValueError("There should be only one ParameterData with the bands distances "
                                      "for pseudo {} and dual {}!".format(pseudo_family,dual))
                elif len(infos_bands_for_pseudo_and_dual)==0:
                    print("     !!! WARNING !!! No ParameterData with the bands distances found "
                                      "for pseudo {} and dual {}!".format(pseudo_family,dual))
                
                try: 
                    bands_distances_params = infos_bands_for_pseudo_and_dual[0].inp.output_parameters.get_inputs_dict()                    
                    bands_distances_params = sorted( [ p  for p in bands_distances_params.itervalues()]  , 
                            key=lambda x:x.inp.output_parameters.inp.bandsdata1.inp.output_band.res.wfc_cutoff)
                
                    band_distances = [ p.get_dict()['results'] for p in bands_distances_params ]
                    #eta_10s = [p['eta_10']*1000.0 + count*offset for p in band_distances] + [0.0 + count*offset] # [meV]
                    #eta_vs  = [p['eta_v']*1000.0 + count*offset for p in band_distances] + [0.0 + count*offset] # [meV]
                    eta_10s = [p['eta_10']*1000.0  for p in band_distances] + [0.0] # [meV]
                    max_10s = [p['max_10']*1000.0 for p in band_distances] + [0.0] # meV
    #                eta_vs  = [p['eta_v']*1000.0  for p in band_distances] + [0.0] # [meV]
    
                    cutoffs_bands = [round(p.inp.output_parameters.inp.bandsdata1.inp.output_band.res.wfc_cutoff/rytoev,1)
                                      for p in bands_distances_params ] + [200.0]
                except IndexError:
                    has_bands = False
                ### END BAND PART
   
                ### DELTA PART
                # Retrieve all info files for a given element and pseudo library (for the Delta factor)   
                infos_for_pseudo = [p for p in infos_for_element if p.get_dict()['pseudo_md5'] == pseudo_md5sum]
                # Retrieve Delta factor for a given element and pseudo library    
                if len(set([p.get_dict()['delta'] for p in infos_for_pseudo if p.get_dict()['delta'] != None])) > 1:
                    raise Exception('     ERROR! More than one delta value found for element={} '
                                    'and pseudo={}!'.format(element,pseudo_family))
                elif set([p.get_dict()['delta'] for p in infos_for_pseudo]) == {None} or  [p.get_dict()['delta'] 
                        for p in infos_for_pseudo] == []:
                    print("     !!! WARNING !!! Element={} and pseudo={} does not have the "
                                    "delta factor.".format(element,pseudo_family))
                    delta = None
                elif element == 'Dy' and pseudo_family == 'RE_Wentz_plus_nitrogen':
                    delta = None
                else:
                    delta = set([p.get_dict()['delta'] for p in infos_for_pseudo if p.get_dict()['delta'] != None]).pop()
                    delta_units = set([p.get_dict()['delta_units'] for p in infos_for_pseudo]).pop()
                    print("     Element={} and pseudo={} has a Delta factor of {} {}".format(element,pseudo_family,
                        round(delta,3),delta_units))
                ### END DELTA PART
    
                ### Calculate Delta1
                    B0_pseudo = set([p.get_dict()['B0'] for p in infos_for_pseudo if p.get_dict()['B0'] != None]).pop()
                    V0_pseudo = set([p.get_dict()['V0'] for p in infos_for_pseudo if p.get_dict()['V0'] != None]).pop()
                    B1_pseudo = set([p.get_dict()['B1'] for p in infos_for_pseudo if p.get_dict()['B1'] != None]).pop()

                    useasymm = False
                    # read reference file
                    if element in rare_earths:
                        ref_file_path = reference_EOS_REN.get_file_abs_path()
                    else:
                        ref_file_path = reference_EOS.get_file_abs_path()

                    try:
                        data_ref = np.loadtxt(ref_file_path, 
                        dtype={'names': ('element', 'V0', 'B0', 'BP'),
                               'formats': ('S2', np.float, np.float, np.float)})
                    except IOError as e:
                        raise IOError("Cannot read the reference file {}: {}"
                                      "".format(ref_file_path,e.message))
                    # build array with the data to be compared to the reference
                    data_tested = np.array([(element,V0_pseudo,B0_pseudo,B1_pseudo)], dtype={
                                            'names': ('element', 'V0', 'B0', 'BP'),
                                            'formats': ('S2', np.float, np.float, np.float),
                                           })
                    eloverlap = list(set(data_tested['element']) & set(data_ref['element']))
                    if not eloverlap:
                        raise ValueError("Element {} is not present in the reference set"
                                          "".format(element))
                    # Delta computation
                    Delta, Deltarel, Delta1 = calcDelta(data_tested, data_ref,
                                        eloverlap, useasymm)
                    Delta1 = Delta1[0]   # Delta1 in meV/atom
                    # print '     Delta1 = {} meV/atom ; Delta = {} meV/atom'.format(Delta1 , Delta[0])
                ### End calculation of Delta1


                if not B0:
                    all_B0s = list(set([p.get_dict()['B0'] for p in infos_for_pseudo if p.get_dict()['B0'] != None]))
                    if len(all_B0s)>1 or len(all_B0s)==0:
                        raise ValueError("There should be one and only one B0 "
                                          "for pseudo {}!".format(pseudo_family))
                    B0 = all_B0s[0]
                    all_B1s = list(set([p.get_dict()['B1'] for p in infos_for_pseudo if p.get_dict()['B1'] != None]))
                    if len(all_B1s)>1 or len(all_B1s)==0:
                        raise ValueError("There should be one and only one B1 "
                                          "for pseudo {}!".format(pseudo_family))
                    B1 = all_B1s[0]
                    all_V0s = list(set([p.get_dict()['V0'] for p in infos_for_pseudo if p.get_dict()['V0'] != None]))
                    if len(all_V0s)>1 or len(all_V0s)==0:
                        raise ValueError("There should be one and only one V0 "
                                          "for pseudo {}!".format(pseudo_family))
                    V0 = all_V0s[0]

                
                # Retrieve all info files for a given element, dual and pseudo library
                if pseudo_family in norm_conserving_pseudo_families:
                    infos_for_dual_nc = [p for p in infos_for_element if p.get_dict()['dual'] == 4.0]   
                    infos_for_pseudo_and_dual = sorted([p for p in infos_for_dual_nc 
                                                 if p.get_dict()['pseudo_md5'] == pseudo_md5sum],
                                                 key=lambda x:x.get_dict()['wfc_cutoff'])
                else:
                    infos_for_pseudo_and_dual = sorted([p for p in infos_for_dual 
                                                 if p.get_dict()['pseudo_md5'] == pseudo_md5sum],
                                                 key=lambda x:x.get_dict()['wfc_cutoff'])

                try:
                    ref_ph_cutoff,ref_ph = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['phonon_frequencies'])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies']!= None])
                except ValueError:
                    print("     !!! WARNING !!! No phonons found "
                                      "for pseudo {} and dual {}!".format(pseudo_family,dual))
                    continue
                if element == 'H' or element == 'I':
                    ref_ph_cutoff,ref_ph = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['phonon_frequencies'][4:])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies']!= None])
                if element == 'Cl' or element == 'N':
                    ref_ph_cutoff,ref_ph = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['phonon_frequencies'][12:])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies']!= None])
                if element == 'F' or element == 'O':
                    ref_ph_cutoff,ref_ph = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['phonon_frequencies'][6:])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies']!= None])
                if element == 'Te' and pseudo_family == 'GBRV_1.2' or element == 'I' and pseudo_family == 'GBRV_1.2':
                    ref_ph_cutoff,ref_ph = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['phonon_frequencies'])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies']!= None and p.get_dict()['wfc_cutoff'] < 200.0 ])
                    
                ref_pressure_cutoff,ref_pressure = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['pressure'])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['pressure']!= None])
                
                has_cohesive = True
                try:
                    ref_cohesive_cutoff,ref_cohesive = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['cohesive_energy'])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['cohesive_energy']!= None])       
                    if element == 'Te' and pseudo_family == 'GBRV_1.2':
                        ref_cohesive_cutoff,ref_cohesive = max([(p.get_dict()['wfc_cutoff'],p.get_dict()['cohesive_energy'])
                        for p in infos_for_pseudo_and_dual if p.get_dict()['cohesive_energy']!= None and p.get_dict()['wfc_cutoff']<200.0])        
                except ValueError:
                    print("     !!! WARNING !!! No Cohesive energies found "
                                      "for pseudo {} and dual {}!".format(pseudo_family,dual))
                    has_cohesive = False
                                                    
                all_Zs = list(set([p.get_dict()['Z'] for p in infos_for_pseudo if p.get_dict()['Z'] != None]))
                if len(all_Zs)>1:
                    raise ValueError("There should be only one Z "
                                      "for pseudo {}!".format(pseudo))
                Z = all_Zs[0]
                
                # TODO: decide what to plot for the phonons
                max_freq = round(max(ref_ph),1)
#                 freqs_ph = [(np.array(p.get_dict()['phonon_frequencies'])-np.array(ref_ph))/max_freq*100+count*offset
#                             for p in infos_for_pseudo_and_dual if p.get_dict()['phonon_frequencies'] != None]
                cutoffs_ph = [p.get_dict()['wfc_cutoff'] for p in infos_for_pseudo_and_dual
                              if p.get_dict()['phonon_frequencies'] != None]
                pressures = [p.get_dict()['pressure'] for p in infos_for_pseudo_and_dual
                             if p.get_dict()['pressure'] != None]
                cutoffs_pressure = [p.get_dict()['wfc_cutoff'] for p in infos_for_pseudo_and_dual
                                    if p.get_dict()['pressure'] != None]
                if has_cohesive:
                    cohesives = [p.get_dict()['cohesive_energy'] for p in infos_for_pseudo_and_dual
                             if p.get_dict()['cohesive_energy'] != None]
                    cutoffs_cohesive = [p.get_dict()['wfc_cutoff'] for p in infos_for_pseudo_and_dual
                                    if p.get_dict()['cohesive_energy'] != None]
                    if element == 'Te' and pseudo_family == 'GBRV_1.2':
                        cohesives = [p.get_dict()['cohesive_energy'] for p in infos_for_pseudo_and_dual
                             if p.get_dict()['cohesive_energy'] != None and p.get_dict()['wfc_cutoff']<200.0]
                        cutoffs_cohesive = [p.get_dict()['wfc_cutoff'] for p in infos_for_pseudo_and_dual
                                    if p.get_dict()['cohesive_energy'] != None and p.get_dict()['wfc_cutoff']<200.0]


                ## DEBUG Absolute mean value for the phonon (with max and min difference) 
                freqs_ph_tot = [np.array(p.get_dict()['phonon_frequencies']) for p in infos_for_pseudo_and_dual 
                                if p.get_dict()['phonon_frequencies'] != None]
                if element == 'H' or element == 'I':
                    freqs_ph_tot = [np.array(p.get_dict()['phonon_frequencies'][4:]) for p in infos_for_pseudo_and_dual 
                                if p.get_dict()['phonon_frequencies'] != None]
                if element == 'Cl' or element == 'N':
                    freqs_ph_tot = [np.array(p.get_dict()['phonon_frequencies'][12:]) for p in infos_for_pseudo_and_dual 
                                if p.get_dict()['phonon_frequencies'] != None]
                if element == 'F' or element == 'O':
                    freqs_ph_tot = [np.array(p.get_dict()['phonon_frequencies'][6:]) for p in infos_for_pseudo_and_dual 
                                if p.get_dict()['phonon_frequencies'] != None]
                if element == 'Te' and pseudo_family == 'GBRV_1.2' or element == 'I' and pseudo_family == 'GBRV_1.2':
                    cutoffs_ph = [p.get_dict()['wfc_cutoff'] for p in infos_for_pseudo_and_dual
                              if p.get_dict()['phonon_frequencies'] != None and p.get_dict()['wfc_cutoff'] < 200.0]
                    freqs_ph_tot = [np.array(p.get_dict()['phonon_frequencies']) for p in infos_for_pseudo_and_dual 
                                if p.get_dict()['phonon_frequencies'] != None and p.get_dict()['wfc_cutoff'] < 200.0]

                if element in elements_low_freq_ph:
                    # in cm^-1
                    if sssp_efficiency:
                        freqs_ph_abs = [math.sqrt(sum(abs(ph-ref_ph)**2)/len(ref_ph)) + count*offset for ph in freqs_ph_tot ]
                        max_diff_ph_abs = [max(abs(ph-ref_ph)) + count*offset for ph in freqs_ph_tot ]
                    else:
                        freqs_ph_abs = [2.*math.sqrt(sum(abs(ph-ref_ph)**2)/len(ref_ph)) + count*offset for ph in freqs_ph_tot ]
                        max_diff_ph_abs = [2.*max(abs(ph-ref_ph)) + count*offset for ph in freqs_ph_tot ]    
                else:
                    # in percentage
                    if sssp_efficiency:
                        freqs_ph_rel = [math.sqrt(sum(abs((ph-ref_ph)/ref_ph)**2)/len(ref_ph))*100.0 + count*offset 
                                    for ph in freqs_ph_tot ]
                        max_diff_ph_rel = [max(abs((ph-ref_ph)/ref_ph))*100. + count*offset for ph in freqs_ph_tot ]
                    else:
                        freqs_ph_rel = [2.*math.sqrt(sum(abs((ph-ref_ph)/ref_ph)**2)/len(ref_ph))*100.0 + count*offset 
                                    for ph in freqs_ph_tot ]
                        max_diff_ph_rel = [2.*max(abs((ph-ref_ph)/ref_ph))*100. + count*offset for ph in freqs_ph_tot ]
                        
                # DEBUG 
                for p in infos_for_pseudo_and_dual:
                    try:
                        #print p.get_dict()['wfc_cutoff'], 'tot_num_freq =', len(p.get_dict()['phonon_frequencies']), \
                        #      'Freqs =', p.get_dict()['phonon_frequencies'][0:18]#, ' max_freq =', \
                        #      p.get_dict()['phonon_frequencies'][-1], ' cohesive =', p.get_dict()['cohesive_energy']  
                         #print 'Bulk k-grid :' , p.inp.output_parameters.get_inputs_dict()['parameters_energy_bulk'].inp.output_parameters.inp.kpoints.get_kpoints_mesh()[0]
                         #print 'Bulk smearing :' , p.inp.output_parameters.get_inputs_dict()['parameters_energy_bulk'].inp.output_parameters.inp.parameters.get_dict()['SYSTEM']['degauss']
                         #print 'Phonon k-grid :', p.inp.output_parameters.get_inputs_dict()['parameters_phonon_bulk'].inp.output_parameters.inp.parent_calc_folder.inp.remote_folder.inp.kpoints.get_kpoints_mesh()[0]   
                         #print 'Phonon smearing :', p.inp.output_parameters.get_inputs_dict()['parameters_phonon_bulk'].inp.output_parameters.inp.parent_calc_folder.inp.remote_folder.inp.parameters.get_dict()['SYSTEM']['degauss']  
                         #print 'Phonon k-grid :', p.inp.output_parameters.get_inputs_dict()['parameters_phonon_bulk'].inp.output_parameters.inp.parameters.get_dict()
                        pass
                    except (KeyError,AttributeError,TypeError):
                        pass    
                # DEBUG
                                
                try:
                    deltas.append(round(delta,3))
                    delta1s.append(round(Delta1,3))
                except TypeError:
                    deltas.append(None)
                    delta1s.append(None)
                Zs.append(Z)
                max_freq = round(max(ref_ph),1)
                max_freqs.append(max_freq)       
                     
                # Stresses
                if B0:
                    delta_pressure_str.append('$\delta V_{P^{\,res}}=$')
                    pressure_unit.append("%")
                else:
                    raise Exception("I cannot calculate the differences of pressure in percentage!")
                    delta_pressure_str.append('$\delta P=$')
                    pressure_unit.append(pressure_units[0])
                # Plot pressures
                if B0:
                    if sssp_efficiency:
                        diff_pressures_rel = [2.*100.*(sssp_utils.get_volume_from_pressure_birch_murnaghan(
                                        ref_pressure-P,V0,B0,B1)/V0-1) + count*offset for P in pressures]
                    else:
                        diff_pressures_rel = [4.*100.*(sssp_utils.get_volume_from_pressure_birch_murnaghan(
                                        ref_pressure-P,V0,B0,B1)/V0-1) + count*offset for P in pressures]
                        
                    
                if not has_bands:
                    pass
                else:
                    for eta_10,max_10,cutoff in zip(eta_10s,max_10s,cutoffs_bands):
                        if eta_10 > 1000.:
                            plt.text(float(cutoff),-offset/3+offset*count,str(int(round(max_10,0))),
                                 color='black',horizontalalignment='center',fontsize=14)
                            plt.text(float(cutoff),offset/3+offset*count+1,str(int(round(eta_10,0))),
                                color='black',horizontalalignment='center',
                                verticalalignment='top',fontsize=14)
                        else:
                            plt.text(float(cutoff),-offset/3+offset*count,str(round(max_10,2)),
                                 color='black',horizontalalignment='center',fontsize=14)
                            plt.text(float(cutoff),offset/3+offset*count+1,str(round(eta_10,2)),
                                color='black',horizontalalignment='center',
                                verticalalignment='top',fontsize=14)
    
                ## Heats of formation (e.g. cohesive energies)
                ## Plot heat of formations
                if has_cohesive:
                    diff_cohesive = [(cohesive-ref_cohesive) + count*offset for cohesive in cohesives]    
                    # diff_cohesive_rel = [(cohesive-ref_cohesive)/ref_cohesive*100. + count*offset for cohesive in cohesives]
                
                if element in  elements_low_freq_ph:
                    l1 = plt.errorbar(cutoffs_ph,freqs_ph_abs,yerr=[np.zeros(len(freqs_ph_abs)), 
                            np.array(max_diff_ph_abs)-np.array(freqs_ph_abs)], ecolor=pseudos_colors_dict[pseudo_dict[pseudo_family]],
                            fmt='o-',capthick=3,capsize=4,color=pseudos_colors_dict[pseudo_dict[pseudo_family]],alpha=0.8,lw=2,ms=10)
                else:        
                    l1 = plt.errorbar(cutoffs_ph,freqs_ph_rel,yerr=[np.zeros(len(freqs_ph_rel)), 
                            np.array(max_diff_ph_rel)-np.array(freqs_ph_rel)], ecolor=pseudos_colors_dict[pseudo_dict[pseudo_family]],
                            fmt='o-',capthick=3,capsize=4,color=pseudos_colors_dict[pseudo_dict[pseudo_family]],alpha=0.8,lw=2,ms=10)
                
                l2, =plt.plot(cutoffs_pressure,diff_pressures_rel,'v-',color=pseudos_colors_dict[pseudo_dict[pseudo_family]],
                         alpha=0.9,lw=2,ms=10,linestyle='--')        

                if has_cohesive:
                    l3, = plt.plot(cutoffs_cohesive,diff_cohesive,'*-',color=pseudos_colors_dict[pseudo_dict[pseudo_family]],
                         alpha=0.9,lw=2,ms=10, linestyle=':')
                                                
                if has_cohesive:                                                
                    plot_lines.append([l1,l2,l3])
                else:
                    plot_lines.append([l1,l2])
                                              
                count += 1
                labels.append(pseudo_dict[pseudo_family])
                plt.axhline(2+offset*(count-1),color=pseudos_colors_dict[pseudo_dict[pseudo_family]],ls='-.',lw=3)  # horizontal line is at y=2
                plt.axhline(-2+offset*(count-1),color=pseudos_colors_dict[pseudo_dict[pseudo_family]],ls='-.',lw=3)
        
                if cutoffs_ph != []:
                    if has_cohesive:
                        max_cutoff = max(max_cutoff,max(cutoffs_ph),max(cutoffs_pressure),max(cutoffs_cohesive))
                    else:
                        max_cutoff = max(max_cutoff,max(cutoffs_ph),max(cutoffs_pressure))
        ypos = []
        ylab = []
        for i in range(count):
            plt.text(max_cutoff+13.5,offset*i,'$\omega_{max}$ = '+str(max_freqs[i])+' {}'.format(phonon_unit.replace('-1','$^{-1}$')),
                horizontalalignment='right',verticalalignment='center',fontsize=14)
            plt.text(max_cutoff+18,offset*i,labels[i]+'\nZ = '+str(Zs[i])+'\n$\Delta$ = '+str(deltas[i])+"\n$\Delta'$ = "+str(delta1s[i]),
                verticalalignment='center',horizontalalignment='center',fontsize=14)
            #plt.text(max_cutoff+18,offset*i, '{}\nZ = {}\n$\Delta$ = {}'.format(labels[i],str(Zs[i]),str(deltas[i])),
            #    verticalalignment='center',horizontalalignment='center',fontsize=14)
            #plt.text(max_cutoff+18,offset*i, '\\begin{align*}  \Delta &= {}  \\\\ \Delta_1 &= {} \end{align*}'.format(labels[i],str(Zs[i]),str(deltas[i]),str(delta1s[i])),
            #    verticalalignment='center',horizontalalignment='center',fontsize=14)
            
            plt.text(28,offset/3+offset*i+1,'$\eta_{10} =$',horizontalalignment='right',
                verticalalignment='top',fontsize=14)
            plt.text(28,-offset/3+offset*i,'$\max \eta_{10} =$',
                horizontalalignment='right',fontsize=14)

            plt.text(max_cutoff+5,offset/3+offset*i+1,"[meV]",
                horizontalalignment='left',verticalalignment='top',fontsize=14)
            plt.text(max_cutoff+5,-offset/3+offset*i,"[meV]",
                horizontalalignment='left',fontsize=14)
                        
#             for j in [-2,-1,0,1,2]:
#                 ypos.append(j+offset*i)
#             for j in ['','','0','','']:
#                 ylab.append(j)
            for j in [-2,0,2]:
                ypos.append(j+offset*i)
            for j in ['','0','']:
                ylab.append(j)
                
        plt.yticks(ypos,ylab,fontsize=14)
        legend = plt.legend(plot_lines[i], [r'$\delta \bar{\omega}$', '$\delta V_{press}$', '$\delta E_{coh}$'], 
                            bbox_to_anchor=(-0.03, 1.0), fontsize=14, frameon=True, markerscale=1.0) 
    
#         if element in elements_low_freq_ph:
#             legend = plt.legend(plot_lines[i], [r'$\delta \bar{\omega}$ [cm$^{-1}$]', '$\delta V_{P^{\,res}}$ [\%]', '$\delta E_{coh}$ [meV/atom]'], 
#                             bbox_to_anchor=(-0.05, 1.0), fontsize=14, frameon=True, markerscale=1.0)
#         else:
#             legend = plt.legend(plot_lines[i], [r'$\delta \bar{\omega}$ [\%]', '$\delta V_{P^{\,res}}$ [\%]', '$\delta E_{coh}$ [meV/atom]'], 
#                             bbox_to_anchor=(-0.05, 1.0), fontsize=14, frameon=True, markerscale=1.0)            
        
        plt.gca().add_artist(legend)
        
        plt.xlabel('Wavefunction cutoff [Ry]; dual = '+str(dual)+
                   ' (PAW/US), dual = 4 (NC); q-point = '+str([round(q,2) for q in qpoint]),fontsize=20)
        if sssp_efficiency:
            plt.ylabel(r'Error w.r.t. ref. wavefunction cutoff (for the SSSP efficiency criteria)',fontsize=20)
        else:
            plt.ylabel(r'Error w.r.t. ref. wavefunction cutoff (for the SSSP precision criteria)',fontsize=20)
        plt.xticks([30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,220,240],
                   ['30','40','50','60','70','80','90','100','110','120','130','140','150','160','170','180','190','200','220','240'],fontsize=14)
        plt.xlim(23,max_cutoff+14)
        plt.ylim(-offset/2.,(len(deltas)-0.5)*offset)
    
        if element == 'F':
            plt.title(element+' (in the crystal structure of SiF4)',fontsize=25)
        elif element == 'Fe' or element == 'Mn':
            plt.title(element+' (the suggested dual is 12)',fontsize=25)
        elif not sssp_efficiency and element == 'Co':
            plt.title(element+' (the suggested dual is 12)',fontsize=25)
        elif element in rare_earths:
            plt.title(element+' (in the crystal structure of nitrides)',fontsize=25)
        else:
            plt.title(element,fontsize=25)
            
        # Put circle at the SSSP pseudopotential/cutoff
        colors = ['#C0C0C0','#FFD700']
        # if sssp_efficiency:
        #     with open(os.path.expanduser(''),'r') as o:
        #         lines = o.readlines()
        #     for line in lines:
        #         line = line.split(' ')
        #         if line[0] == element and line[1] != '??':
        #             from matplotlib.patches import Ellipse
        #             ax = plt.gca()
        #             try:
        #                 ellipse = Ellipse(xy=(float(line[1]), labels.index(line[3].split('\n')[0])*offset),
        #                       width=2, height=2, edgecolor=colors[0], fc='None', lw=5)
        #                 ax.add_patch(ellipse)
        #             except ValueError:
        #                 pass
        # else:
        #     with open(os.path.expanduser('~/Dropbox/sssp/sssp_v1.0/sssp_accuracy.txt'),'r') as o:
        #         lines = o.readlines()
        #     for line in lines:
        #         line = line.split(' ')
        #         if line[0] == element and line[1] != '??':
        #             from matplotlib.patches import Ellipse
        #             ax = plt.gca()
        #             try:
        #                 ellipse = Ellipse(xy=(float(line[1]), labels.index(line[3].split('\n')[0])*offset),
        #                       width=2, height=2, edgecolor=colors[1], fc='None', lw=5)
        #                 ax.add_patch(ellipse)
        #             except ValueError:
        #                 pass

        if sssp_efficiency:
            ps_eff   = sssp_eff_table['{}'.format(element)]['pseudopotential']
            cut_eff  = sssp_eff_table['{}'.format(element)]['cutoff']

            ax = plt.gca()
            try:
                ellipse = Ellipse(xy=(cut_eff, labels.index(ps_eff)*offset),
                          width=2, height=2, edgecolor=colors[0], fc='None', lw=5)
                ax.add_patch(ellipse)
            except ValueError:
                pass

        else:
            ps_prec   = sssp_prec_table['{}'.format(element)]['pseudopotential']
            cut_prec  = sssp_prec_table['{}'.format(element)]['cutoff']

            ax = plt.gca()
            try:
                ellipse = Ellipse(xy=(cut_prec, labels.index(ps_prec)*offset),
                          width=2, height=2, edgecolor=colors[0], fc='None', lw=5)
                ax.add_patch(ellipse)
            except ValueError:
                pass
              
        plt.savefig(element+'_'+str(dual)+'_conv_patt.png')
        plt.close()

                
