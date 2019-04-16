import sys 
from aiida.orm.user import User     
from aiida.backends.djsite.db import models
from sssp_tools import sssp_utils
from sssp_tools.sssp_utils import pseudo_families_libraries_dict  as pseudo_dict
from matplotlib import pylab as plt
import numpy as np
import matplotlib

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

ParameterData = DataFactory('parameter')
UpfData = DataFactory('upf')

'''
Script to generate the png files with the pseudopotential equation of state for each element.

'''


# Name of the AiiDA group to query from the SSSP data
groups=Group.query(name__startswith='info_',name__endswith='_sssp1')
groups=Group.query(name__startswith='La_',name__endswith='_sssp')
   
rare_earths = ['La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
infos = ParameterData.query(dbgroups__pk__in=[_.pk for _ in groups])

# Take all the elements in the groups 
elements = set(infos.filter(dbattributes__key='element').values_list('dbattributes__tval',flat=True))
# User-specified elements
elements = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',  
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
            'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te', 
            'I','Xe','Cs','Ba','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','Rn',
            'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
#elements = ['La']

warnings = []
  
for element in elements:
    print '*****************'
    print 'Element = {}'.format(element)
    print '*****************'

    pseudo_families = [
                    'pslib.0.3.1_PBE_US',
                    'pslib.0.3.1_PBE_PAW',
                    'pslib.1.0.0_PBE_US',
                    'pslib.1.0.0_PBE_PAW',
                    'pslib.1.0.0_PBE_US_low_acc',
                    'pslib.1.0.0_PBE_PAW_low_acc',
                    'GBRV_1.2',
                    'GBRV_1.4',
                    'GBRV_1.5',
                    'SG15',
                    'SG15_1.1',
                    'THEOS',
                    'Goedecker',

#                     'RE_Wentz',
#                     'RE_Wentz_plus_nitrogen',
#                     'RE_pslib.1.0.0_PBE_US_plus_nitrogen',
#                     'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen',
                   ]
    #pseudo_families = ['SG15','SG15_1.1']

    if element in rare_earths:
        pseudo_families = [
                            'RE_Wentz_plus_nitrogen',
                            'RE_pslib.1.0.0_PBE_US_plus_nitrogen',
                            'RE_pslib.1.0.0_PBE_PAW_plus_nitrogen',
                            #'RE_Vander_plus_nitrogen',
                            #'RE_GBRV_plus_nitrogen',
                            #'RE_SG15_plus_nitrogen'
                           ]

        
    infos_for_element = infos.filter(dbattributes__key='element',dbattributes__tval=element)
    
    pseudos_md5 = set(infos.filter(dbattributes__key='pseudo_md5').filter(dbattributes__key='element',
                    dbattributes__tval=element).values_list('dbattributes__tval',flat=True))


    for pseudo_family in pseudo_families:
        
        pseudo_family_has_element = False
        pseudo_family_group = UpfData.get_upf_group(pseudo_family)
        for pseudo in pseudo_family_group.nodes:
            if element == pseudo.element:
                pseudo_md5sum = pseudo.md5sum
                pseudo_family_has_element = True
    
        print ''
        print '---> Pseudopotential library = {}'.format(pseudo_family)
        print ''

        if not pseudo_family_has_element:
            print '     No pseudopotential for this library!'

        if pseudo_family_has_element:
            
            infos_for_pseudo = [p for p in infos_for_element if p.get_dict()['pseudo_md5'] == pseudo_md5sum]
            # Retrieve all cutoffs and duals present in the info files (for a given element and pseudo library)  
#            cutoffs_and_duals = list(set([(p.get_dict()['wfc_cutoff'],p.get_dict()['dual']) for p in infos_for_pseudo]))
    
            pseudo_name = UpfData.query(dbattributes__in=models.DbAttribute.objects.filter(key='md5', 
                                                tval=pseudo_md5sum)).first().filename
            pseudo_family = UpfData.query(dbattributes__in=models.DbAttribute.objects.filter(key='md5', 
                                                tval=pseudo_md5sum)).first().get_upf_family_names()
            pseudo_family = [p for p in pseudo_family if p not in ['BaTiO_test','AuIn2_ONCV1.0-1.1','SG15_1.0-1.1']][0]                                    
                
            # Retrieve Delta factor for a given element and pseudo library    
            if len(set([p.get_dict()['delta'] for p in infos_for_pseudo if p.get_dict()['delta'] != None])) > 1:
                raise Exception('     ERROR! More than one delta value found for element={} '
                                'and pseudo={}!'.format(element,pseudo_name))
            elif set([p.get_dict()['delta'] for p in infos_for_pseudo]) == {None} or [p.get_dict()['delta'] 
                    for p in infos_for_pseudo] == []:
                print("     WARNING! Element={} and pseudo={} does not have the "
                                "delta factor.".format(element,pseudo_name))
                warnings.append("WARNING! Element={} and pseudo_family={} does not have the "
                                "delta factor.".format(element,pseudo_family))
            else:
                delta = set([p.get_dict()['delta'] for p in infos_for_pseudo if p.get_dict()['delta'] != None]).pop()
                delta_units = set([p.get_dict()['delta_units'] for p in infos_for_pseudo]).pop()
                print("     Element={} and pseudo={} has a Delta factor of {} {}".format(element, 
                        pseudo_name, round(delta,3), delta_units))
                    
                params_data = [p for p in infos_for_pseudo if p.get_dict()['delta'] != None][0]
                params_delta = params_data.inp.output_parameters.inp.parameters_delta
                eos_inline = params_delta.inp.output_parameters
    
                energies = []
                volumes = []
                avg_magnetization = 0.0
                for k,v in eos_inline.get_inputs_dict().iteritems():
                    if k=='parameters':
                        pass                      
                    else:
                        if type(v)==ParameterData:
                            pw_calc = v.inp.output_parameters
                            #print("--- Absolute magnetization = {} {}".format(pw_calc.res.absolute_magnetization,
                            #                                           pw_calc.res.absolute_magnetization_units))
                            #print("--- Total magnetization = {} {}".format(pw_calc.res.total_magnetization,
                            #                                           pw_calc.res.total_magnetization_units))
                            #print pw_calc.pk
                            #   
                            #avg_magnetization +=  pw_calc.res.total_magnetization
                            number_of_atoms = pw_calc.res.number_of_atoms
                            energies.append(pw_calc.res.energy/number_of_atoms)  # eV/atoms
                            volumes.append(pw_calc.res.volume/number_of_atoms)  # Angstrom^3/atoms
                            # print 'cutoff: {} Ry,  k-grid: {}, degauss: {} Ry'.format(pw_calc.inp.parameters.get_dict()['SYSTEM']['ecutwfc'],pw_calc.inp.kpoints.get_kpoints_mesh()[0],pw_calc.inp.parameters.get_dict()['SYSTEM']['degauss'])
                   
                #avg_magnetization = avg_magnetization/7. 
                #print   avg_magnetization          
                # Fit params for EOS
                eos_fit_params = eos_inline.out.output_parameters.get_dict()
                E0 = eos_fit_params['Birch_Murnaghan_fit_parameters']['E0']
                V0 = eos_fit_params['Birch_Murnaghan_fit_parameters']['V0']
                B0 = eos_fit_params['Birch_Murnaghan_fit_parameters']['B0']
                B1 = eos_fit_params['Birch_Murnaghan_fit_parameters']['B1']
    
                    
                # Wien2k data
                reference_EOS_file = eos_inline.get_inputs_dict()['reference_EOS_file']
                ref_file_path = reference_EOS_file.get_file_abs_path()
                data_ref = np.loadtxt(ref_file_path, 
                                      dtype={'names': ('element', 'V0', 'B0', 'BP'),
                                             'formats': ('S2', np.float, np.float, np.float)})
                count = 0
                for el in data_ref['element']:
                    if el == element:
                        E0_wien2k = E0
                        V0_wien2k = data_ref['V0'][count]
                        B0_wien2k = data_ref['B0'][count]
                        B1_wien2k = data_ref['BP'][count]
                        break
                    count += 1
    
                plt.plot(volumes,energies,'go')
                new_x = np.arange(min(volumes), max(volumes), 0.01)
                plt.plot(new_x, sssp_utils.birch_murnaghan(new_x, E0, V0, B0/160.2176487, B1), 'g-', label='Fit')
                plt.plot(new_x, sssp_utils.birch_murnaghan(new_x, E0_wien2k, V0_wien2k, B0_wien2k/160.2176487, 
                                                           B1_wien2k), color='black', label='WIEN2k')
                #plt.text(max(volumes)-1.0,max(energies)-0.010, 'Avg. total magnetization = {} Bohr mag. / cell \n'
                #         'Delta factor = {} meV/atom'.format(round(avg_magnetization,3),
                #        [p.get_dict()['delta'] for p in infos_for_pseudo if p.get_dict()['delta'] != None][0]),
                #         verticalalignment='center', horizontalalignment='center',fontsize=12)
                try:
                    plt.text(reduce(lambda x, y: x + y, volumes) / len(volumes),
                        reduce(lambda x, y: x + y, energies) / len(energies), 
                        '$\Delta$ = {} meV/atom'.format(round(delta,3),delta_units),
                        verticalalignment='center', horizontalalignment='center',fontsize=14)
                except KeyError:
                    pass

                plt.xlabel('Volume [A$^3$/atom]')
                plt.ylabel('Energy [eV/atom]')
                plt.title('EOS for '+element+' ('+pseudo_dict[pseudo_family]+')')
                plt.legend()
                plt.savefig(element+'_'+pseudo_dict[pseudo_family]+'_eos.png')
                plt.close()
    
            print ''
            print ''

print warnings 
