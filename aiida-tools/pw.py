# -*- coding: utf-8 -*-
from aiida.orm.workflow import Workflow
from aiida.orm import DataFactory, Node, Code, CalculationFactory, Group, Computer
from aiida.common.datastructures import calc_states
from aiida.common.example_helpers import test_and_get_code
from sssp-protocol.aiida-tools.quantumespresso import helpers
from aiida.common.exceptions import AiidaException,ValidationError,InternalError,NotExistent
from aiida.orm.data.array.kpoints import _default_epsilon_length,_default_epsilon_angle
from aiida.orm.calculation.inline import make_inline,optional_inline
from math import exp,ceil
import os
import numpy as np
try:
    from aiida.backends.utils import get_authinfo
except ImportError:
    from aiida.execmanager import get_authinfo

__copyright__ = u"Copyright (c), This file is part of the AiiDA-EPFL Pro platform. For further information please visit http://www.aiida.net/. All rights reserved"
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file."
__version__ = "0.1.0"
__authors__ = "Nicolas Mounet, Andrea Cepellotti, Giovanni Pizzi."


UpfData = DataFactory('upf')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
BandsData = DataFactory('array.bands')
StructureData = DataFactory('structure')
PwCalculation = CalculationFactory('quantumespresso.pw')

_default_beta_mix_from_QE = 0.7
_default_mode_mix_from_QE = 'plain'
_default_ndim_mix_from_QE = 8
_default_nspin_from_QE = 1
_default_startmag_from_QE = 0.
_default_noncolin_from_QE = False
_default_electron_maxstep_from_QE = 100
_default_press_conv_thr_from_QE = 0.5 # Kbar
_default_press_from_QE = 0. # Kbar
_default_occupations_from_QE = None
_default_degauss_from_QE = 0.
_default_diagonalization_from_QE = 'david'
_default_wf_collect_from_QE = False
_diagonalization_algorithms_from_QE = ['david','cg']
# various mixing schemes to try in case of slow convergence
#_mixing_schemes_to_try = [('plain',0.3,16),('local-TF',0.3,16),('TF',0.3,16)]


class PwrestartWorkflowError(AiidaException):
    """
    Raised when the Pwrestart wf failed on an error.
    self.error_type gives the kind of error.
    """
    def __init__(self, *args, **kwargs):
        # Call the base class constructor with the parameters it needs
        super(PwrestartWorkflowError, self).__init__(*args)

        self.error_type = kwargs.pop('error_type', '')
        if kwargs:
            raise ValueError("Unknown parameters passed to PwrestartWorkflowError")


class PwbandsrestartWorkflowError(AiidaException):
    """
    Raised when the Pwbandsrestart wf failed on an error.
    self.error_type gives the kind of error.
    """
    def __init__(self, *args, **kwargs):
        # Call the base class constructor with the parameters it needs
        super(PwbandsrestartWorkflowError, self).__init__(*args)

        self.error_type = kwargs.pop('error_type', '')
        if kwargs:
            raise ValueError("Unknown parameters passed to PwbandsrestartWorkflowError")


def get_parallelization_parameters(pw_init_calc,max_num_machines,
                                   target_time,calculation='scf',n_kpoints=None,
                                   power_law_time_vs_nbands=(exp(-16.1951988),1.22535849)):
    """
    Guess an optimal choice of parallelzation parameters.
    :param pw_init_calc: an initial pw calculation (only initialization is sufficient),
        to get number of k-points, of electrons, of spins, fft grids, etc.
    :param max_num_machines: the maximum allowed number of nodes to be used
    :param target_time: time the calculation should take finally for the user
        (in seconds)
    :param calculation: kind of calculation to be performed ('scf','nscf',
        'bands','relax','md','vc-relax','vc-md')
    :param power_law_time_vs_nbands: list or tuple with 2 numbers giving the 
        fit parameters for a power law expressing the single-CPU time to do 
        1 scf step, for 1 k-point, 1 spin and 1 small box of the fft grid, 
        as a function of number of electrons, in the form:
            normalized_single_CPU_time = A*n_elec^B
        where A is the first number and B the second.
        Default values were obtained on piz-dora (CSCS) in 2015, on a set of 
        4370 calculations (with a very rough fit).
    :return: suggested parallelization parameters: number of machines, 
        number of procs per machine, number of k-points pools, 
        and total estimated time of the calc (for the user, in seconds).
        
    .. note:: If there was an out-of-memory problem during the initial
        calculation, the number of machines is increased.
    """
    from fractions import gcd # function to get the greatest common divisor
    
    default_num_mpiprocs_per_machine = pw_init_calc.get_computer().get_default_mpiprocs_per_machine()
    pw_init_calc_output = pw_init_calc.out.output_parameters.get_dict()
    pw_init_calc_input = pw_init_calc.inp.parameters.get_dict()
    if n_kpoints is None:
        n_kpoints = pw_init_calc_output['number_of_k_points']
    n_spins = pw_init_calc_input['SYSTEM'].get(
                            'nspin',_default_nspin_from_QE)
    fft_grid = pw_init_calc_output['fft_grid']
    if calculation in ['bands','nscf']:
        n_scf = 1
    else:
        n_scf = pw_init_calc_input.get('ELECTRONS',{}).get(
                            'electron_maxstep',_default_electron_maxstep_from_QE)
        if calculation in ['relax','md','vc-relax','vc-md']:
            n_scf *= 6 # assuming 6 relaxation steps (rule of thumb)
    n_bands = pw_init_calc_output['number_of_bands']
    
    # compute an estimate single-CPU time
    singleCPU_time = np.prod(fft_grid) * n_spins * n_kpoints * n_scf *\
                    power_law_time_vs_nbands[0] * n_bands**power_law_time_vs_nbands[1]
    
    # the number of nodes is the maximum number we can use that is dividing n_kpoints
    num_machines = max([m for m in range(1,max_num_machines+1) if n_kpoints % m == 0])
    if ( num_machines == 1 and n_kpoints > 6 and max_num_machines > 1
         and singleCPU_time/default_num_mpiprocs_per_machine > target_time ) :
        # in that case we retry, making n_kpoints even
        num_machines = max([m for m in range(1,max_num_machines+1)
                            if (n_kpoints+1) % m == 0])
    
    # now we will try to decrease the number of procs. per machine (by not more 
    # than one fourth) until we manage to get an efficient plane wave parallelization
    # (i.e. number of procs per pool dividing the third dimension of the fft grid)
    num_mpiprocs_per_machine = default_num_mpiprocs_per_machine
    successful = False
    while num_mpiprocs_per_machine >= 0.75*default_num_mpiprocs_per_machine:
        if n_kpoints % num_machines != 0:
            npools = num_machines
        else:
            npools = num_machines * gcd(num_mpiprocs_per_machine,n_kpoints/num_machines)
        if fft_grid[2] % num_mpiprocs_per_machine/(npools/num_machines) == 0:
            successful = True
            break
        num_mpiprocs_per_machine -= 1
    
    if not successful:
        num_mpiprocs_per_machine = default_num_mpiprocs_per_machine
        if n_kpoints % num_machines != 0:
            npools = num_machines
        else:
            npools = num_machines * gcd(num_mpiprocs_per_machine,n_kpoints/num_machines)
        
    # increase the number of machines in case of memory problem during initialization
    # TODO: make it more general and less dependent on the scheduler exact message
    if pw_init_calc.get_scheduler_error() and 'OOM' in pw_init_calc.get_scheduler_error():
        num_machines = max([i for i in range(num_machines,max_num_machines+1)
                            if i%num_machines==0])
    
    estimated_time = singleCPU_time/(num_mpiprocs_per_machine*num_machines)
    
    return num_machines,num_mpiprocs_per_machine,npools,estimated_time

@make_inline
def copy_pw_density_files_inline(parameters,remote_folder):
    """
    Inline calculation to copy the charge density and spin polarization files
    from a pw calculation
    :param parameters: ParameterData object with a dictionary of the form
        {'destination_directory': absolute path of directory where to put the files,
         'destination_computer_name': name of the computer where to put the file
                                      (if absent or None, we take the same
                                      computer as that of remote_folder)
         }
    :param remote_folder: the remote folder of the pw calculation
    :return: a dictionary of the form
        {'density_remote_folder': RemoteData_object}
    """
    RemoteData = DataFactory('remote')
    params_dict = parameters.get_dict()
    computer_dest_name = params_dict.get('destination_computer_name',None)
    if computer_dest_name:
        computer_dest = Computer.get(computer_dest_name)
    else:
        computer_dest = remote_folder.get_computer()
    t_dest = get_authinfo(computer=computer_dest,
                          aiidauser=remote_folder.get_user()).get_transport()
    dest_dir = params_dict['destination_directory']
    # get the uuid of the parent calculation
    calc = remote_folder.inp.remote_folder
    calcuuid = calc.uuid
    t_source = get_authinfo(computer=remote_folder.get_computer(),
                            aiidauser=remote_folder.get_user()).get_transport()
    source_dir = os.path.join(remote_folder.get_remote_path(),
                              PwCalculation._OUTPUT_SUBFOLDER)

    with t_source,t_dest:
        # get the density files name
        density_dir = os.path.join(source_dir,
                                   "{}.save".format(PwCalculation._PREFIX))
        t_source.chdir(density_dir)
        content_list = t_source.listdir()
        density_files = [f for f in content_list 
                         if (any([f.startswith(_) for _ in ('charge-density',
                                                             'magnetization',
                                                             'spin-polarization')])
                             and f.find('old.')==-1)]
        
        # zip the density files (keeping the original files)
        for density_file in density_files:
            _,stdout,stderr = t_source.exec_command_wait(" ".join(
                ["gzip","-c",density_file,">","{}.gz".format(density_file)]))
            # -c option is to keep original file and output on std output
            if stderr:
                raise InternalError("Error while compressing the density "
                                    "file(s): {}".format(stderr))
        # zip also the paw file, if present
        # the paw file is not at the same location before and after QE 6.0...
        QEversion = calc.out.output_parameters.get_dict()['creator_version']
        if QEversion >= '6.0':
            paw_file = os.path.join("{}.save".format(PwCalculation._PREFIX),"paw.txt")
        else:
            paw_file = "{}.paw".format(PwCalculation._PREFIX)
        t_source.chdir(source_dir)
        if t_source.isfile(paw_file):
            _,stdout,stderr = t_source.exec_command_wait(" ".join(
                ["gzip","-c",paw_file,">","{}.gz".format(paw_file)]))
            # -c option is to keep original file and output on std output
            density_files.append(paw_file)
            if stderr:
                raise InternalError("Error while compressing the paw "
                                    "file(s): {}".format(stderr))
        
        # build the destination folder
        t_dest.chdir(dest_dir)
        # we do the same logic as in the repository and in the working directory,
        # i.e. we create the final directory where to put the file splitting the
        # uuid of the calculation
        t_dest.mkdir(calcuuid[:2], ignore_existing=True)
        t_dest.chdir(calcuuid[:2])
        t_dest.mkdir(calcuuid[2:4], ignore_existing=True)
        t_dest.chdir(calcuuid[2:4])
        t_dest.mkdir(calcuuid[4:])
        t_dest.chdir(calcuuid[4:])
        final_dest_dir = t_dest.getcwd()
        
        # copy the zipped files and remove them from the source
        density_file_paths = [os.path.join(source_dir,
            "{}.save".format(PwCalculation._PREFIX),"{}.gz".format(f))
            if f != paw_file else os.path.join(source_dir,"{}.gz".format(f))
            for f in density_files]
        for density_file_gz in density_file_paths:
            if t_dest._machine == t_source._machine:
                t_source.copy(density_file_gz,final_dest_dir)
            else:
                t_source.copy_from_remote_to_remote(t_dest,density_file_gz,
                                                    final_dest_dir)
            t_source.remove(density_file_gz)
    
    density_remote_folder = RemoteData(computer=computer_dest,
                                       remote_path=final_dest_dir)
    
    return {'density_remote_folder': density_remote_folder}

@make_inline
def build_pw_remote_folder_from_density_files_inline(parameters,
                                                     density_remote_folder,
                                                     retrieved_folder):
    """
    Inline calculation to create a usable remote folder from which a
    pw calculation can be started. It copies the charge density (and spin
    polarization) file(s) to a remote folder, unzip them, and copy also
    there the xml data file from the retrieved folder of a pw
    calculation.
    :param parameters: ParameterData object with a dictionary of the form
        {'destination_directory': absolute path of the directory where to 
                                  put the output remote folder
                                  (if absent, we use the default workdir
                                   in the destination computer),
         'destination_computer_name': name of the computer where to put 
                                      the output remote folder
                                      (if absent or None, we use the same
                                      computer as that of density_remote_folder),
         }
    :param density_remote_folder: the remote folder of the charge density files
    :param retrieved_folder: the retrieved folder of the pw calculation
        to be used to get the xml data file

    :return: a dictionary of the form
        {'pw_remote_folder': RemoteData_object containing the unzipped
                             density files and the xml data file
         }
    
    .. note:: the charge density and the retrieved folder can in 
        principle come from 2 different pw calculations.
    """
    from aiida.common.utils import get_new_uuid
    
    def copy_and_unzip(t_source,t_dest,source_dir,source_file,dest_dir,
                       dest_file=''):
        """
        Copy file source_dir/source_file, from remote machine defined in 
        t_source transport, to the destination dest_dir/dest_file 
        (or to dest_dir/ if dest_file is '') in remote machine defined 
        in t_dest transport; then unzip the file.
        Case when source and destination are or are not on the same 
        computer are both handled.
        """
        source_path = os.path.join(source_dir,source_file)
        dest_path = os.path.join(dest_dir,dest_file)
        # copy
        if t_dest._machine == t_source._machine:
            t_source.copy(source_path,dest_path)
        else:
            t_source.copy_from_remote_to_remote(t_dest,source_path,dest_path)
        # unzip
        t_dest.chdir(dest_dir)
        file_to_unzip = dest_file if dest_file else source_file
        _,stdout,stderr = t_dest.exec_command_wait(" ".join(["gunzip",
                                                        file_to_unzip]))
        if stderr:
            raise InternalError("Error while uncompressing the file {}: {}"
                                 "".format(file_to_unzip,stderr))
        return

    RemoteData = DataFactory('remote')
    params_dict = parameters.get_dict()
    t_source = get_authinfo(computer=density_remote_folder.get_computer(),
                            aiidauser=density_remote_folder.get_user()
                            ).get_transport()
    source_dir = density_remote_folder.get_remote_path()
    computer_dest_name = params_dict.get('destination_computer_name',None)
    if computer_dest_name:
        computer_dest = Computer.get(computer_dest_name)
    else:
        computer_dest = density_remote_folder.get_computer()
    authinfo_dest = get_authinfo(computer=computer_dest,
                                 aiidauser=density_remote_folder.get_user())
    t_dest = authinfo_dest.get_transport()
    
    # uuid to be used to create the working directory of the output
    # remote folder
    uuid = get_new_uuid()
    
    with t_source,t_dest:
        # building the remote destination directory
        dest_dir = params_dict.get('destination_directory',
                                   authinfo_dest.get_workdir().format(
                                   username=t_dest.whoami()))
        t_dest.chdir(dest_dir)
        
        # we create the final directory where to put the files splitting
        # the uuid as if it was a remote folder from a calculation with 
        # this uuid
        t_dest.mkdir(uuid[:2], ignore_existing=True)
        t_dest.chdir(uuid[:2])
        t_dest.mkdir(uuid[2:4], ignore_existing=True)
        t_dest.chdir(uuid[2:4])
        t_dest.mkdir(uuid[4:])
        t_dest.chdir(uuid[4:])
        remote_folder_dir = t_dest.getcwd()
        t_dest.mkdir(PwCalculation._OUTPUT_SUBFOLDER)
        t_dest.chdir(PwCalculation._OUTPUT_SUBFOLDER)
        t_dest.mkdir("{}.save".format(PwCalculation._PREFIX))
        t_dest.chdir("{}.save".format(PwCalculation._PREFIX))
        final_dest_dir = os.path.join(t_dest.getcwd(),"")
        # the final "" is to get a path to a directory

        # copy the density files (and the paw file if present)
        paw_files = ["{}.paw.gz".format(PwCalculation._PREFIX),"paw.txt.gz"]
        density_files = t_source.listdir(source_dir)
        for density_file in density_files:
            if density_file not in paw_files:
                # copy and unzip charge density file(s)
                copy_and_unzip(t_source,t_dest,source_dir,density_file,
                               final_dest_dir,dest_file='')
            else:
                # PAW (becsum) file
                # There are two possibilities: either it's PREFIX.paw (before QE 6.0)
                # or PREFIX.save/paw.txt (after QE 6.0, included)
                # For now we take into account these two possibilies and copy the
                # same file (being either one of the two) in BOTH locations.
                copy_and_unzip(t_source,t_dest,source_dir,density_file,
                               os.path.join(final_dest_dir,'..'),
                               dest_file=paw_files[0])
                copy_and_unzip(t_source,t_dest,source_dir,density_file,
                               final_dest_dir,dest_file=paw_files[1])
        
        # now copy the xml data file from the local repository
        xml_file = retrieved_folder.get_abs_path(os.path.join(
                        PwCalculation._DATAFILE_XML_BASENAME))
        t_dest.put(xml_file,final_dest_dir)
        
    pw_remote_folder = RemoteData(computer=computer_dest,
                                  remote_path=remote_folder_dir)
    
    return {'pw_remote_folder': pw_remote_folder}

@make_inline
def get_bands_and_occupations_inline(remote_folder,pw_output_parameters):
    """
    Produces a BandsData with all the bands and their occupations,
        from the last iteration of a PW calc.
    :param remote_folder: output remote folder of a PW calculation.
    :param pw_output_parameters: output parameters from the same Pw calc.
    :return: a dictionary of the form
        {'output_occupations': ArrayData with the bands occupations,
         'output_parameters': ParameterData with some warnings}
    """
    from aiida.parsers.plugins.quantumespresso.raw_parser_pw import parse_pw_xml_output
    from aiida.common.folders import SandboxFolder
    from aiida.parsers.plugins.quantumespresso import QEOutputParsingError
    
    BandsData = DataFactory('array.bands')
    
    # get the pw output parameters, to get number of bands, of k-points
    # and of spin components
    pw = remote_folder.inp.remote_folder
    if pw.pk != pw_output_parameters.inp.output_parameters.pk:
        raise ValueError("remote_folder and pw_output_parameters should"
                         " come from the same PW calculation!")

    pw_output_dict = pw_output_parameters.get_dict()
    warnings = []
    try:
        n_kpts = pw_output_dict['number_of_k_points']
        n_bands = pw_output_dict['number_of_bands']
        n_spins = pw_output_dict['number_of_spin_components']
    except KeyError:
        warnings.append("Cannot find number of k-points, of bands and/or spins"
                        " -> occupations might be wrong")
    
    # get the transport to the remote folder, and the full path to the
    # bands directory
    t_source = get_authinfo(computer=remote_folder.get_computer(),
                            aiidauser=remote_folder.get_user()
                            ).get_transport()
    source_dir = os.path.join(remote_folder.get_remote_path(),
                              pw._OUTPUT_SUBFOLDER,
                              '{}.save'.format(pw._PREFIX))
    
    # read the pw output file (to be able to use the parsing function)
    try:
        with open(pw.out.retrieved.get_abs_path(pw._DATAFILE_XML_BASENAME)) as f:                                               
            data=f.read()
    except OSError:
        warnings.append("Cannot find {} in retrieved folder".format(pw._DATAFILE_XML_BASENAME))
        return {'output_parameters': ParameterData(dict={'warnings': warnings})}
    
    result_dict = {}
    with SandboxFolder() as sandbox, t_source:
        t_source.get(source_dir,sandbox.abspath)
        try:
            p_dict,s_dict,b_dict = parse_pw_xml_output(data,
                dir_with_bands=os.path.join(sandbox.abspath,
                '{}.save'.format(pw._PREFIX)))
        except QEOutputParsingError as e:
            warnings.append('xml parsing failed with message "{}"'.format(e.message))
        else:
            warnings.extend(p_dict['xml_warnings'])
            if 'occupations' in b_dict and b_dict['occupations']:
                if not(np.array(b_dict['occupations']).shape==(n_spins,n_kpts,n_bands) or
                        (n_spins==1 and np.array(b_dict['occupations']).shape==(n_kpts,n_bands))):
                    warnings.append("ocupations shape {} do not correspond "
                                    "to {} (coming from output parameters) "
                                    "-> something is probably wrong".format(
                                    np.array(b_dict['occupations']).shape,
                                    (n_spins,n_kpts,n_bands)))
                # get the kpoints
                kpoints_from_input = pw.inp.kpoints
                try:
                    kpoints_from_input.get_kpoints()
                    kpoints_for_bands = kpoints_from_input
                except AttributeError:
                    kpoints_for_bands = pw.out.output_kpoints
                try:
                    kpoints_from_input.get_kpoints()
                    kpoints_for_bands.labels = kpoints_from_input.labels
                except (AttributeError,ValueError,TypeError):
                    # AttributeError: no list of kpoints in input
                    # ValueError: labels from input do not match the output 
                    #      list of kpoints (some kpoints are missing)
                    # TypeError: labels are not set, so kpoints_from_input.labels=None
                    pass
                    
                # get the bands occupations
                # correct the occupations of QE: if it computes only one component,
                # it occupies it with half number of electrons
                try:
                    b_dict['occupations'][1]
                    the_occupations = b_dict['occupations']
                except IndexError:
                    the_occupations = 2.*np.array(b_dict['occupations'][0])
                # get the bands energies
                try:
                    b_dict['bands'][1]
                    bands_energies = b_dict['bands']
                except IndexError:
                    bands_energies = b_dict['bands'][0]
                
                bands_data = BandsData()
                bands_data.set_kpointsdata(kpoints_for_bands)
                bands_data.set_bands(bands_energies,
                                     units = b_dict['bands_units'],
                                     occupations = the_occupations)
                result_dict['output_band'] = bands_data
            else:
                warnings.append("No occupations found")
    
    result_dict ['output_parameters'] = ParameterData(dict={'warnings': warnings})
    
    return result_dict

@optional_inline
def update_nbands_inline(**kwargs):
    """
    Obtain updated parameters when the number of bands
    are not enough for a pw calc.
    :param pw_output_band: bands (with occupations) from a pw calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :param settings: some additional settings, in particular
        {'max_upper_occupation_allowed': the maximum allowed upper occupation
                                         (default = 0.01)} 
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData, or {}
                               if no update needed,
         'output_parameters' : ParameterData with a dict of the form
            {'max_of_highest_band_occupations': maximum occ. of the highest band}
         }
    :raise TypeError: when no update on the number of bands is needed
        (to avoid storing anything for nothing)
    """
    pw_output_band = kwargs['pw_output_band']
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()
    
    settings = kwargs.pop('settings',{})
    if isinstance(settings,ParameterData):
        settings = settings.get_dict()
    
    max_upper_occupation_allowed = settings.get('max_upper_occupation_allowed',0.01)
    
    bands, occupations = pw_output_band.get_bands(also_occupations=True)
    # if only one spin, add one more dimension to the arrays
    if len(bands.shape)==2:
        bands = [bands]
        occupations = [occupations]
    # sort the bands by energy
    the_bands, the_occupations = np.array(zip(*[
        zip(*[zip(*sorted(zip(bands_each_k,occ_each_k),key=lambda x: x[0]))
        for bands_each_k,occ_each_k in zip(bands_each_spin,occ_each_spin)])
        for bands_each_spin,occ_each_spin in zip(bands,occupations)]))
    highest_band_occupations = np.array([[occ_each_k[-1] for occ_each_k in occ_each_spin]
                                   for occ_each_spin in the_occupations])
    
    res_dict = {'output_parameters': ParameterData(dict={
                    'max_of_highest_band_occupations': np.max(highest_band_occupations)})}
    
    if np.max(highest_band_occupations)>=max_upper_occupation_allowed:
        updated_nbnd = int(round(the_occupations.shape[-1]+max(4,0.2*the_occupations.shape[-1])))
        updated_parameters = helpers.update_nested_dict(previous_parameters,
                                        {'SYSTEM': {'nbnd': updated_nbnd}})
        res_dict['updated_parameters'] = ParameterData(dict=updated_parameters)
    else:
        # if store is False, we output an empty dictionary, while if store
        # is True we raise a TypeError on purpose, to prevent any storing
        # when no update is needed
        # TODO: do something cleaner
        res_dict['updated_parameters'] = {}
    
    return res_dict

# The next inline functions do not all explicitely use the pw
# output parameters given in input; they are only used for linking purposes
# (the pw output is used BEFORE calling these functions).
# TODO: make cleaner functions that uses all their inputs - maybe a 
# general inline function that does what's in all of them, plus outputting
# necessary stuff like the message to report, the 'has_finished' and/or
# 'has_failed' flags, a flag to say if we output the structure, an 
# error message, etc.
@optional_inline
def update_mixing_inline(**kwargs):
    """
    Update beta mixing.
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_parameters = kwargs['pw_output_parameters']
    pw_calc_input = pw_output_parameters.inp.output_parameters.inp.parameters.get_dict()
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()
    
    mixing_beta = pw_calc_input.get('ELECTRONS',{}).get(
                    'mixing_beta',_default_beta_mix_from_QE)
    #mixing_mode = pw_calc_input.get('ELECTRONS',{}).get(
    #                'mixing_mode',_default_mode_mix_from_QE)
    #mixing_ndim = pw_calc_input.get('ELECTRONS',{}).get(
    #                'mixing_ndim',_default_ndim_mix_from_QE)
    #try:
    #    ind = _mixing_schemes_to_try.index((mixing_mode,
    #                        mixing_beta,mixing_ndim))
    #except ValueError:
    #    ind = -1
    #try:
    #    mixing_scheme = _mixing_schemes_to_try[ind+1]
    #except IndexError:
    #    mixing_scheme = (mixing_mode,mixing_beta,mixing_ndim)
    # change mixing scheme in updated parameters
    #update_params['parameters']['ELECTRONS'].update(
    #    {k: v for k,v in zip(['mixing_mode','mixing_beta','mixing_ndim'],mixing_scheme)})
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'ELECTRONS': {'mixing_beta': mixing_beta*0.8}})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_nspin_inline(**kwargs):
    """
    Update nspin and starting magnetization parameters, to launch a
    non-magnetic calculation.
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :param settings: some additional settings, in particular
        {'nspin': updated value for nspin (default = 1),
         'starting_magnetization': updated value for starting magnetization (default=0.)} 
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    settings = kwargs.pop('settings',{})
    if isinstance(settings,ParameterData):
        settings = settings.get_dict()
    
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'SYSTEM': {'nspin': settings.get('nspin',1),
                                        'starting_magnetization': settings.get('starting_magnetization',0.)}})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_degauss_inline(**kwargs):
    """
    Update degauss parameter, to possibly solve the Fermi energy flipping
    issue for some insulators.
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :param settings: some additional settings, in particular
        {'degauss': updated value for degauss [Ry],
         'smearing': (optional) updated value for smearing} 
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    settings = kwargs.pop('settings',{})
    if isinstance(settings,ParameterData):
        settings = settings.get_dict()
    
    system_dict = {'degauss': settings['degauss']}
    if 'smearing' in settings:
        system_dict.update({'smearing': settings['smearing']})
    
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'SYSTEM': system_dict})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_diag_inline(**kwargs):
    """
    Update diagonalization algorithm.
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_parameters = kwargs['pw_output_parameters']
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    pw_calc_input = pw_output_parameters.inp.output_parameters.inp.parameters.get_dict()
    
    diag_algo = pw_calc_input['ELECTRONS'].get(
                    'diagonalization',_default_diagonalization_from_QE)
    try:
        new_diag_algo = _diagonalization_algorithms_from_QE[1 -
                    _diagonalization_algorithms_from_QE.index(diag_algo)]
    except ValueError:
        new_diag_algo = 'cg'

    updated_parameters = helpers.update_nested_dict(previous_parameters,
                        {'ELECTRONS': {'diagonalization': new_diag_algo}})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_max_seconds_inline(**kwargs):
    """
    Update max seconds in CONTROL namelist.
    :param pw_output_parameters: output parameters of a PW calc.
    :param settings: If present, ParameterData with a dict of the form
        {'max_seconds': total wall time that should not be exceeded}
        If it's not there, we use the value from the input of the pw calc.
    .. note:: we always multiply this value by 0.97
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_parameters = kwargs['pw_output_parameters']
    pw_calc_input = pw_output_parameters.inp.output_parameters.inp.parameters.get_dict()
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    settings = kwargs.pop('settings',{})
    if isinstance(settings,ParameterData):
        settings = settings.get_dict()

    max_sec = settings.get('max_seconds',pw_calc_input['CONTROL']['max_seconds'])
    
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'CONTROL': {'max_seconds': int(0.97*max_sec)}})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_starting_magnetization_inline(**kwargs):
    """
    Update starting magnetization.
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_parameters = kwargs['pw_output_parameters']
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    start_mag_dict = helpers.get_starting_magnetization_pw(
                                        pw_output_parameters.get_dict())

    updated_parameters = helpers.update_nested_dict(previous_parameters,
                                    {'SYSTEM': start_mag_dict})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_startingpotwfc_inline(**kwargs):
    """
    Update startingwfc and startingpot input parameters for a charge
    density restart.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    previous_parameters = kwargs['previous_parameters']
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()
                           
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'ELECTRONS': {'startingpot': 'file',
                                           'startingwfc': 'atomic+random'}})
    
    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_press_inline(**kwargs):
    """
    Update pressure in CELL namelist.
    :param pw_output_structure: output structure of a PW calc.
    :param settings: ParameterData with a dict of the form
        {'target_press': target pressure in kbar}
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update (if any)
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_structure = kwargs['pw_output_structure']
    settings = kwargs['settings']
    previous_parameters = kwargs.pop('previous_parameters',{})
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()

    if isinstance(settings,ParameterData):
        settings = settings.get_dict()
    
    target_pressure = settings['target_pressure']
    
    updated_parameters = helpers.update_nested_dict(previous_parameters,
                            {'CELL': {'press': target_pressure}})

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def update_params_for_bandstructure_inline(**kwargs):
    """
    Update pw parameters for a band structure calculation (number of bands,
    and max_seconds - which is suppressed).
    :param pw_output_parameters: output parameters of a PW calc.
    :param previous_parameters: ParameterData or dictionary (the latter
        only if store is False) with the parameters to update
    :param settings: if any, dictionary, or ParameterData with a dict, of the form
        {'number_of_bands': (optional) integer,
         'number_of_bands_nelec_units': (optional) integer (ratio #bands/#nelec)}
    :return : a dictionary of the form
        {'updated_parameters': the updated ParameterData}
    """
    pw_output_parameters = kwargs['pw_output_parameters']
    previous_parameters = kwargs['previous_parameters']
    pw_calc_input = pw_output_parameters.inp.output_parameters.inp.parameters.get_dict()
    pw_calc_output = pw_output_parameters.get_dict()
    
    settings = kwargs.pop('settings',{})
    if isinstance(settings,ParameterData):
        settings = settings.get_dict()
    
    if isinstance(previous_parameters,ParameterData):
        previous_parameters = previous_parameters.get_dict()
    
    # suppress max_seconds (will be re-initialized latter)
    if 'max_seconds' in previous_parameters.get('CONTROL',{}):
        _ = previous_parameters['CONTROL'].pop('max_seconds')
        
    try:
        num_bands = settings['number_of_bands']
    except KeyError:
        try:
            mult_factor = settings['number_of_bands_nelec_units']
            num_elec = pw_calc_output['number_of_electrons']
            num_bands= int(num_elec * mult_factor)
        except KeyError:
            # update the number of bands (take 25% more than half the number
            # of electrons, or at least 4 more bands)
            if pw_calc_output['smearing_method']:
                num_bands = None
            else:
                num_elec = pw_calc_output['number_of_electrons']
                # to add spin-orbit case
                ncol_flag = pw_calc_input.get('SYSTEM',{}).get('noncolin',_default_noncolin_from_QE)
                if ncol_flag:
                    num_bands = int(num_elec)+max(8,int(0.25*num_elec))
                else:
                    num_bands = int(num_elec/2)+max(4,int(0.125*num_elec))

    if num_bands is not None:
        updated_parameters = helpers.update_nested_dict(previous_parameters,
                                {'SYSTEM': {'nbnd': num_bands}})
    else:
        updated_parameters = previous_parameters

    return {'updated_parameters': ParameterData(dict=updated_parameters)}

@optional_inline
def build_kpoints_inline(structure,parameters):
    """
    Build the k-points (uniform mesh or path) from a structure.
    :param parameters: ParameterData with dict of the form
        {'distance_kpoints_in_mesh': distance between adjacent k-pts along the
                             crystal axis (1/ang) - builds a k-pts mesh,
         'distance_kpoints_in_dispersion': distance between adjacent k-pts
                             along the path (1/ang) - builds a k-pts path,
         'offset': (mesh only) offset the k-points mesh (3 floats between 0 ad 1),
         'force_parity': (mesh only) True to force an even number of k-points in each direction,
         'kpoints_path': (path only) user defined path of k-points
                         (see KpointsData.set_kpoints_path method for details)
                         (default=None, i.e. automatic high-sym. path determination) 
         'epsilon_length': (path only) length tolerance for Bravais lattice determination,
         'epsilon_angle': (path only) angle tolerance for Bravais lattice determination}
    :param structure: a structure
    :return: a dictionary of the form
        {'output_kpoints': k-points}
    """
    if isinstance(parameters,ParameterData):
        parameters = parameters.get_dict()

    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    
    if 'distance_kpoints_in_mesh' in parameters:
        kpoints.set_kpoints_mesh_from_density(parameters['distance_kpoints_in_mesh'],
                                              offset=parameters['offset'],
                                              force_parity=parameters['force_parity'])
    else:
        input_dict = {'value': parameters.get("kpoints_path",None),
                      'kpoint_distance': parameters["distance_kpoints_in_dispersion"]}
        if 'epsilon_length' in parameters:
            input_dict['epsilon_length'] = parameters["epsilon_length"]
        if 'epsilon_angle' in parameters:
            input_dict['epsilon_angle'] = parameters["epsilon_angle"]

        kpoints.set_kpoints_path(**input_dict)

    return {'output_kpoints': kpoints}


class PwWorkflow(Workflow):
    """
    Launch energy and bands calculations with Quantum-Espresso - Pwscf,
    with restart management (failure cases handled) and a convergence loop on
    a vc-relax/vc-md on the volume and/or pressure (thus updating the G-vectors
    for each new run).
    k-points can be set using a k-points distance, meaning the distance between
    adjacent k-points along a reciprocal axis, is forced to be lower than the
    this distance. In this case and for a vc-relax/vc-md calculation,
    the k-points grid will be updated for each new run.
    
    The parameters of the workflow can be found below (default values are indicated between
    parentheses, and if the parameter is required or not - note that the
    parameters required for pw don't need to be there if the workflow is started
    from the bands calculation i.e. when the key 'pw_calculation' is present).
    
    .. note:: when 'automatic_parallelization' is present, it overrides the
    'calculation_set' (for the main calc, the final scf calc and the band calc).    
        
    params = {
        'codename' (required for pw):       name of the code to be used,
        'pseudo_family' (required for pw):  name of the pseudo family to be used,
        'structure':                        a previously stored StructureData object,
        'calculation_set':                  dictionary with resources, walltime, etc.
                                            - each key is any NAME that can be set to 
                                            a PW calculation with the command 
                                            pw_calc.set_NAME(...). For example:
                {'resources':{'num_machines': 1},
                 'max_wallclock_seconds': 100,
                 'custom_scheduler_commands': "#SBATCH -A, --account=MY_ACCOUNT"
                 }
                                            It needs to be set with at least the
                                            'resources' and 'max_wallclock_seconds'
                                            keys if 'automatic_parallelization' is
                                            not in the 'input' key (see below),
        
        'kpoints':                          a KpointsData object. If not present,
                                            then a distance between adjacent k-points
                                            is used to generate a regular mesh (see 
                                            below in 'input'),
                                             
        'parameters' (required for pw):     a dictionary (or a ParameterData -
                                            then provenance is kept down
                                            to the calculations - TO BE USED WITH CAUTION 
                                            as it can generate a HUGE graph 
                                            if there are lots of restarts)
                                            with the PW calculation input 
                                            parameters, e.g. :
                  {
                    'CONTROL': {
                        'tstress': True,
                    },
                    'SYSTEM': {
                        'ecutwfc': 40.,
                        'ecutrho': 320.,
                    },
                    'ELECTRONS': {
                        'conv_thr': 1.e-10,
                    }
                  }
        
        'settings':                         a dictionary with additional settings for
                                            the PW calc., e.g. the number of pools
                                            to be used:
                  {'cmdline':['-nk','8']}
        
        'goup_name':                        name of a group where to put the
                                            resulting pw calculation,
                                            
        'input':                            some general input parameters for 
                                            the workflow:
            {
            'clean_workdir' (False):              triggers the automatic cleaning
                                                  of scratch directories of all
                                                  calculations except the last one
                                                  (WARNING: it does not seem to
                                                  work very well)
            'directory_for_density_files':        path to the remote directory
                                                  where to put the charge density
                                                  files (and, in case, the .paw file),
            'max_restarts' (5):                  the maximum number of restarts,
            'max_upper_occupation_allowed'(0.01): (used only with smearing) the maximum
                                                  allowed occupations on the upper band:
                                                  if a calculation with smearing ends up 
                                                  with a higher occupation than that on
                                                  the upper band, at any k-point, then
                                                  we relaunch with a higher number of bands,
            'finish_with_scf' (True):             True to re-do an scf calculation after the
                                                  convergence is reached (note: only 
                                                  for vc-relax and vc-md calculations),
            'distance_kpoints_in_mesh (0.2):      when kpoints are not defined,
                                                  generate a regular k-points mesh
                                                  using this distance (adjacent k-pts
                                                  along reciprocal axes are separated
                                                  at least by this value), 
            'offset_kpoints_mesh ([0., 0., 0.]):  when kpoints are generated from
                                                  a distance, we can offset them
                                                  by this offset (floats between 0 and 1
                                                  - i.e. [0.5,0.5,0.5] is a half shifted grid)
            'force_parity_kpoints_mesh' (False):  if True, the k-points mesh
                                                  contains only even numbers
                                                  (except in non-periodic directions
                                                  where it is always 1),
            'volume_convergence_threshold' (0.1): relative volume convergence threshold
                                                  for a vc-relax or vc-md relaxation,
            'pressure_convergence' (False):       True to enforce also a pressure
                                                  convergence (on top of the volume
                                                  convergence), for vc-relax/vc-md,
            'relaxation_scheme' ('scf'):          relaxation_scheme - it overrides
                                                  'parameters' -> 'CONTROL' -> 'calculation',
            'automatic_parallelization':          switch on the automatic 
                                                  parallelization setter (i.e.
                                                  number of pools/procs/nodes 
                                                  and max cpu time are determined
                                                  by an algorithm using some
                                                  parameters obtained from 
                                                  a first initialization calculation).
                                                  It should contain a dictionary
                                                  of the form:
                   {
                    'max_wall_time_seconds': max_time_seconds,
                    'target_time_seconds': target_time_seconds,
                    'max_num_machines': max_num_machines
                   }
            },
            
        'remote_folder':  a RemoteData with the remote_folder output of a 
                          previously run calculation. Used to restart a PW workflow
                          (including scf or relaxation) from a previous PW calc.
        
        'charge_density_folder': the remote data containing previously saved 
                                 charge density files. Used to restart a PW workflow
                                 (including scf or relaxation) from a previous PW calc.
                                 when the full scratch is not available but the charge 
                                 density was saved.
        'retrieved_folder':      retrieved data from a previously run PW calculation.
                                 To be used together with 'charge_density_folder',
                                 Not mandatory - if not present an initialization will be launched -
                                 but it can help (convergence in one iteration) when present
        
        'pw_calculation': a PwCalculation representing a previously run calculation
                          (FINISHED). If present, only the bands calculation is 
                          launched, and the keys required for the pw calculations
                          are not anymore required.
                
        'finalscf_parameters_update':       a dictionary with parameters to update
                                            the pw parameters that initially come 
                                            from the initial calculation, e.g.:
                          {
                           'SYSTEM': 
                               {
                                 'use_all_frac': False,
                                },
                           'ELECTRONS':
                               {
                                'conv_thr': 1e-10
                                },
                           }
        
        # then can come all the parameter keys beginning with 'band_', that are for 
        # the bands structure calculation:
        
        'band_calculation_set':  dictionary with resources, walltime, etc.
                                 - each key is any NAME that can be set to 
                                 a PW calculation with the command 
                                 pw_calc.set_NAME(...). For example:
                {'resources':{'num_machines': 1},
                 'max_wallclock_seconds': 100,
                 'custom_scheduler_commands': "#SBATCH -A, --account=MY_ACCOUNT"
                 },
                NOTE: if it is not set, and if band_input -> automatic_parallelization
                is also not present, the corresponding parameters from the parent PW calc.
                will be used (but CONTROL-> max_seconds will not be set accordingly).
                                            
        'band_settings':                    a dictionary with additional settings for
                                            the PW calc., e.g. the number of pools
                                            to be used (note: you don't need
                                            to specify in these settings
                                            also_bands=True - it is done 
                                            automatically in the workflow):
                  {'cmdline':['-nk','8']},
                                            If no number of pools is present here,
                                            the number of pools is determined 
                                            automatically (using a first 'fake' 
                                            bands calculation to get n_kpoints).
                  
        'band_kpoints':                     a KpointsData object with the k-points
                                            to be used for the bands calculation.
                                            If not present, an automatic dispersion
                                            along high-symmetry lines is generated
                                            (see 'band_input' below),
                                            
        'band_group_name':                   name of a group where to put the
                                            resulting bands structure,
        
        'band_input':                       general parameters for the bands workflow:
                    {
                       'automatic_parallelization':       switch on the automatic 
                                                          parallelization setter (i.e.
                                                          number of pools/procs/nodes 
                                                          and max cpu time are determined
                                                          by an algorithm using some
                                                          parameters obtained from 
                                                          a first initialization calculation).
                                                          It should contain a dictionary
                                                          of the form:
                               {
                                'max_wall_time_seconds': max_time_seconds,
                                'target_time_seconds': target_time_seconds,
                                'max_num_machines': max_num_machines
                               }
                       'max_restarts' (5):                the maximum number of restarts,
                       'clean_workdir' (False):
                                triggers the automatic cleaning of scratch directories
                                of all calculations except the pw_calculation result
                                (WARNING: it does not seem to work very well),
                       'number_of_bands' (default QE number of bands when smearing is applied,
                                          or n_elec+max(8,0.25*n_elec) for noncolin calc.,
                                          or n_elec/2+max(4,0.125*n_elec) otherwise):
                                number of bands to be computed,
                       'number_of_bands_nelec_units':
                                number of bands in units of the number of electrons
                       'distance_kpoints_in_dispersion' (0.01): 
                                distance between two consecutive k-points in
                                the dispersion, calculated in crystal coordinates,
                       'distance_kpoints_in_mesh' : if present, build a regular
                                kpoints mesh based on the specified distance (in 1/ang)
                                between 2 consecutive k-points along reciprocal axes
                       'offset_kpoints_mesh' ([0.,0.,0.]): if 'distance_kpoints_in_mesh'
                                is specified, use this value to offset the mesh,
                       'force_parity_kpoints_mesh' (False): if True and
                                'distance_kpoints_in_mesh' is specified, the k-points mesh
                                contains only even numbers (except in non-periodic
                                directions where it is always 1),
                       'kpoints_path' (default: high-symmetry path from KpointsData class): 
                                a kpoints path, e.g. [('G','X')],
                       'threshold_length_for_Bravais_lat' (default from KpointsData class): 
                                threshold on lengths comparison, used
                                to get the bravais lattice info (to generate the
                                high-symmetry path).
                       'threshold_angle_for_Bravais_lat' (default from KpointsData class): 
                                threshold on angles comparison, used
                                to get the bravais lattice info (to generate the
                                high-symmetry path).
                       
                    },
                    
        'band_parameters_update':  a dictionary (or a ParameterData) 
                                   with parameters to update
                                   the bands pw parameters that initially come 
                                   from the parent pw calculation, e.g.:
                          {
                           'ELECTRONS':
                               {
                                'diagonalization':'cg',
                                'conv_thr': 1e-10
                                },
                           }
    }


    """
    # Default values
    _clean_workdir = False
    _volume_conv_thr = 0.1
    _pressure_conv = False
    _finish_with_scf = True
    #_final_scf_remove_use_all_frac = False # By default, do not remove 'use_all_frac' even if present (but ph wf should do it)
    _default_calc = 'scf'
    _default_distance_kpoints_in_mesh = 0.2
    _default_offset_kpoints_mesh = [0., 0., 0.]
    _default_force_parity_kpoints_mesh = False
    _default_epsilon_length = _default_epsilon_length
    _default_epsilon_angle = _default_epsilon_angle
    _default_distance_kpoints_in_dispersion = 0.01
    
    def __init__(self,**kwargs):
        super(PwWorkflow, self).__init__(**kwargs)
    
    @Workflow.step
    def start(self):
        """
        Starting step only verifies the input parameters
        """
        self.append_to_report("Checking PW input parameters")
        
        # define the mandatory keywords and the corresponding description to be 
        # printed in case the keyword is missing, for the PW parameters
        mandatory_pw_keys = [('codename',basestring,'the PW codename'),
                             ('structure',StructureData,"the structure (a previously stored StructureData object)"),
                             ('pseudo_family',basestring,'the pseudopotential family'),
                             #('kpoints',KpointsData,'A KpointsData object with the kpoint mesh used by PWscf'),
                             #('calculation_set',dict,'A dictionary with resources, walltime, ...'),
                             #('input',dict,"a dictionary with the workflow parameters"),
                             ('parameters',(ParameterData,dict),"A dictionary or ParameterData with the PW input parameters"),
                             ]
        
        mandatory_band_keys = [#('band_calculation_set',dict,'A dictionary with resources, walltime, ...'),
                               ]
        
        # retrieve and check the PW parameters
        params = self.get_parameters()

        try:
            pw_calculation = params['pw_calculation']
        except KeyError:
            pass
        else:
            self.add_result('pw_calculation', pw_calculation)
            self.append_to_report("Using existing pw calc {}".format(
                    pw_calculation.pk))
            helpers.validate_keys(params, mandatory_band_keys)
            self.next(self.run_bands)
            return

        self.append_to_report("Starting a new pw calc")
        helpers.validate_keys(params,mandatory_pw_keys)
        
        # check the code
        test_and_get_code(params['codename'], 'quantumespresso.pw',
                          use_exceptions=True)
        
        # Note: I'm not checking extensively all the keys used by the workflow
        self.next(self.run_pw)
        
    @Workflow.step
    def run_pw(self):
        # for vc-relax or vc-md calculations, 
        # this step loops on itself until the cell volume
        # does not move by more than main_params['volume_conv_thr']
        # between two successive calculations
        from copy import deepcopy

        # retrieve PW parameters
        main_params = self.get_parameters()
        
        try:
            main_params['input']
        except KeyError:
            main_params['input'] = {}
        try:
            main_params['input']['relaxation_scheme']
        except KeyError:
            main_params['input']['relaxation_scheme'] = self._default_calc
        
        # we presume that we don't need to launch a PwrestartWorkflow
        has_to_launch = False
        previous_error_type = self.get_attributes().get('error_type','')
        # check if this is the case
        # get the list of Pwrestart workflows launched up to now
        wf_pw_list = list(self.get_step(self.run_pw).get_sub_workflows(
                                                            ).order_by('ctime'))
        if not wf_pw_list:
            has_to_launch = True
            pwrestart_params = deepcopy(main_params)
        
        if wf_pw_list:
            last_subworkflow = wf_pw_list[-1]
            pwrestart_params = last_subworkflow.get_parameters()
            # if some parameters (mixing beta, diagonalization algorithm, etc.)
            # were changed during restart workflow, apply the changes also
            # to the new workflow, apart from several parameters (see
            # below: nspin, smearing)
            update_params = {k.replace('update_',''): v for k,v in 
                             last_subworkflow.get_attributes().items()
                             if k.startswith('update_')}
            
            try:
                last_pw_calc = last_subworkflow.get_result('pw_calculation')
                if main_params.get('input',{}).get('relaxation_scheme',
                                    self._default_calc) in ['vc-relax','vc-md']:
                    # update parameters with output structure
                    pwrestart_params['structure'] = last_pw_calc.out.output_structure
            
            except ValueError:
                # the sub-workflow is in error state -> get the kind of error
                error_type = last_subworkflow.get_attributes().get('error_type','')
                if error_type==previous_error_type:
                    # if the sub-wf already encountered the same error type, we stop
                    raise InternalError('Sub-wf {} got twice the error {}'.format(
                                        last_subworkflow.pk,
                                        last_subworkflow.get_report()[-2]))
                # otherwise, try to get the output structure
                try:
                    pwrestart_params['structure'] = last_subworkflow.get_result('structure')
                    self.add_attribute('error_type',error_type)
                    has_to_launch = True
                except ValueError:
                    raise InternalError('Sub-wf {} did not output any structure,'
                                        ' failed with error {} '.format(
                                        last_subworkflow.pk,
                                        last_subworkflow.get_report()[-2]))

            if last_subworkflow.get_attributes().get('changed_nspin',False):
                if (len(wf_pw_list)>1 and wf_pw_list[-2].get_attributes(
                    ).get('changed_nspin',False)):
                    self.append_to_report(" ... vc-relax + spin-polarization"
                                          " got trouble twice, we will "
                                          "put NOT back the spin-polarization ... ")
                else:
                    # if there was a change from spin-polarized to unpolarized calculation
                    # during the last restart workflow, AND NOT DURING THE LAST BUT ONE,
                    # it should be changed back
                    update_input_dict = {'store': isinstance(main_params['parameters'],ParameterData),
                                         'settings': ParameterData(
                        dict={'nspin': helpers.get_from_parameterdata_or_dict(
                                    main_params['parameters'],'SYSTEM',default={}).get(
                                    'nspin',_default_nspin_from_QE),
                              'starting_magnetization': helpers.get_from_parameterdata_or_dict(
                                    main_params['parameters'],'SYSTEM',default={}).get(
                                    'starting_magnetization',_default_startmag_from_QE)
                              }),
                                         }
                    if 'parameters' in update_params and update_params['parameters']:
                        update_input_dict['previous_parameters'] = \
                                                update_params['parameters']
                    res_dict = update_nspin_inline(**update_input_dict)
                    update_params['parameters'] = res_dict['updated_parameters']
                    self.append_to_report(" ... needs another vc-relax run putting "
                                          " back the spin polarization ... ")
                    has_to_launch = True

            if last_subworkflow.get_attributes().get('changed_degauss',False):
                if (len(wf_pw_list)>1 and wf_pw_list[-2].get_attributes(
                    ).get('changed_degauss',False)):
                    self.append_to_report(" ... we already changed back degauss, "
                                          "we will not do it again ... ")
                    raise InternalError("Sub-wf {} failed a second time"
                                    " with the original degauss "
                                    "(report: {})".format(last_subworkflow.pk,
                                    last_subworkflow.get_report()[-2]))
                else:
                    # if there was a change from spin-polarized to unpolarized calculation
                    # during the last restart workflow, AND NOT DURING THE LAST BUT ONE,
                    # it should be changed back
                    update_input_dict = {'store': isinstance(main_params['parameters'],ParameterData),
                                         'settings': ParameterData(dict={
                                                    'degauss': helpers.get_from_parameterdata_or_dict(
                                    main_params['parameters'],'SYSTEM',default={}).get(
                                    'degauss',_default_degauss_from_QE)}),
                                         }
                    if 'parameters' in update_params and update_params['parameters']:
                        update_input_dict['previous_parameters'] = \
                                                update_params['parameters']
                    res_dict = update_degauss_inline(**update_input_dict)
                    update_params['parameters'] = res_dict['updated_parameters']
                    self.append_to_report(" ... needs another vc-relax run putting "
                                          " back the original degauss ... ")
                    has_to_launch = True

            # check the change of volume between input and output structures
            input_volume = last_subworkflow.get_parameter('structure').get_cell_volume()
            output_volume = pwrestart_params['structure'].get_cell_volume()
            volume_conv_thr = main_params.get('input',{}).get(
                           'volume_convergence_threshold',self._volume_conv_thr)
            # check pressure, in case one wants to converge it as well
            pressure_conv_thr = helpers.get_from_parameterdata_or_dict(
                main_params['parameters'],'CELL',default={}).get(
                            'press_conv_thr',_default_press_conv_thr_from_QE)
            delta_pressure = 0.
            target_pressure = helpers.get_from_parameterdata_or_dict(
                    pwrestart_params['parameters'],'CELL',default={}).get(
                    'press',_default_press_from_QE)
            if main_params.get('input',{}).get('pressure_convergence',self._pressure_conv):
                # rescale target pressure with new volume (last pressure is
                # computed on final structure). In the end what we try to impose
                # is the product P*V = P_initial*V_initial.
                target_pressure *= input_volume / output_volume
                try:
                    delta_pressure = abs(target_pressure - 10.*np.trace(
                        last_pw_calc.out.output_parameters.get_dict()['stress'])/3.)
                    # factor of 10. is for conversion from GPa (in pw output) to KBar (in pw input)
                except (UnboundLocalError,NameError):
                    delta_pressure = 1e50
            
            if ( abs((input_volume-output_volume)/input_volume) > volume_conv_thr
                 or delta_pressure > pressure_conv_thr):
                # update parameters with output structure
                if ( target_pressure != 0. and delta_pressure > 0):
                    # change the pressure imposed according to the change in volume
                    update_input_dict = {'pw_output_structure': pwrestart_params['structure'],
                                         'settings': ParameterData(
                                            dict={'target_pressure': target_pressure}),
                                         'store': isinstance(main_params['parameters'],ParameterData)}
                    try:
                        update_input_dict['pw_output_parameters'] = last_pw_calc.out.output_parameters
                    except (UnboundLocalError,NameError):
                        pass
                    if 'parameters' in update_params and update_params['parameters']:
                        update_input_dict['previous_parameters'] = \
                                                update_params['parameters']
                    res_dict = update_press_inline(**update_input_dict)
                    update_params['parameters'] = res_dict['updated_parameters']
                has_to_launch = True
                    
            # apply all the changes to the parameters of the workflow
            pwrestart_params = helpers.update_nested_dict(pwrestart_params,update_params)
        
        # deal with kpoints
        try:
            _ = main_params['kpoints']
        except KeyError:
            kpoints_params = ParameterData(dict={
                'distance_kpoints_in_mesh': main_params.get('input',{}).get(
                                        'distance_kpoints_in_mesh',
                                        self._default_distance_kpoints_in_mesh),
                'offset': main_params.get('input',{}).get('offset_kpoints_mesh',
                                        self._default_offset_kpoints_mesh),
                'force_parity': main_params.get('input',{}).get('force_parity_kpoints_mesh',
                                        self._default_force_parity_kpoints_mesh)})
            res_dict = build_kpoints_inline(structure=pwrestart_params['structure'],
                                            parameters=kpoints_params,
                                            store=True)
            pwrestart_params['kpoints'] = res_dict['output_kpoints']
            
        # drop any kind of remote folder in Pwrestart wf params
        # if the structure has changed
        if pwrestart_params['structure'].pk!=main_params['structure'].pk:
            pwrestart_params.pop('remote_folder',None)
            pwrestart_params.pop('charge_density_folder',None)
            pwrestart_params.pop('retrieved_folder',None)
        
        if has_to_launch:
            # Launch the PW computation (using restart workflow)
            wf_pw = PwrestartWorkflow(params=pwrestart_params)
            wf_pw.store()
            self.append_to_report("Launching Pwrestart computation (wf pk: {}) "
                "with volume={}{}".format(wf_pw.pk,pwrestart_params['structure'].get_cell_volume(),
                " and target pressure={} Kbar".format(
                helpers.get_from_parameterdata_or_dict(pwrestart_params['parameters'],'CELL',default={}).get('press',_default_press_from_QE))
                if main_params.get('input',{}).get('pressure_convergence',self._pressure_conv) else ""))
            self.attach_workflow(wf_pw)
            wf_pw.start()
            if (main_params.get('input',{}).get('relaxation_scheme',
                                    self._default_calc) in ['vc-relax','vc-md']
                or main_params.get('input',{}).get('finish_with_scf',self._finish_with_scf)):
                # loop on itself only if variable cell calculation
                # (which induces volume change -> change in G vectors)
                # or if we want to do a final scf calculation
                self.next(self.run_pw)
            else:
                self.next(self.results_pw)
                
        else:
            if 'final_convergence_threshold' in main_params['input']:
                self.append_to_report("WARNING: 'final_convergence_threshold' ignored,"
                                      "specify it in 'finalscf_parameters_update' "
                                      "-> 'ELECTRONS' -> 'conv_thr'")
            if 'final_scf_remove_use_all_frac' in main_params['input']:
                self.append_to_report("WARNING: 'final_scf_remove_use_all_frac' ignored,"
                                      "specify it in 'finalscf_parameters_update' "
                                      "-> 'SYSTEM' -> 'use_all_frac' -> False")
                
            finish_with_scf = main_params.get('input',{}).get('finish_with_scf',self._finish_with_scf)
            if finish_with_scf:
                # we finish with an scf calculation for which the parameters
                # can be customized at will
                #for k in main_params.keys():
                #    # take all the parameters that begins with 'finalscf_' and use
                #    # them to update the parameters
                #    if k.startswith('finalscf_') and k!='finalscf_parameters_update':
                #        v = main_params.pop(k)
                #        new_k = k[9:] # remove finalscf_
                #        pwrestart_params[new_k] = v
                # redefine structure and relaxation scheme
                pwrestart_params['structure'] = last_pw_calc.out.output_structure
                pwrestart_params['input']['relaxation_scheme'] = 'scf'
                # possibly update parameters
                update_dict=main_params.get('finalscf_parameters_update',{})
                if update_dict:
                    pwrestart_params['parameters'] = \
                        helpers.update_nested_dict(pwrestart_params['parameters'],update_dict)
                    if (isinstance(pwrestart_params['parameters'],ParameterData)
                        and pwrestart_params['parameters']._to_be_stored):
                        pwrestart_params['parameters'] = pwrestart_params['parameters'].get_dict()
                        self.append_to_report("WARNING: there is a 'ParameterData' object in "
                                              "'parameters' but a dict in "
                                              "'finalscf_parameters_update' -> "
                                              "provenance on parameters cannot be kept "
                                              "for the final scf calculation")
                # create and launch subworkflow
                wf_pw = PwrestartWorkflow(params=pwrestart_params)
                wf_pw.store()
                self.append_to_report("Launching final PW scf computation "
                                      "(wf pk: {})".format(wf_pw.pk))
                self.attach_workflow(wf_pw)
                wf_pw.start()
            
            #final_scf_remove_use_all_frac = params.get('input',{}).get('final_scf_remove_use_all_frac',
            #                                                           self._final_scf_remove_use_all_frac)
            #if finish_with_scf:
            #    # we finish with an scf calculation
            #    params['structure'] = last_pw_calc.out.output_structure
            #    params['input']['relaxation_scheme'] = 'scf'
            #    if 'final_convergence_threshold' in params['input'].keys():
            #        try:
            #            params['parameters']['ELECTRONS']['conv_thr'] = params['input']['final_convergence_threshold']
            #        except KeyError:
            #            params['parameters']['ELECTRONS'] = {'conv_thr': params['input']['final_convergence_threshold']}
            #    # delete relaxation namelists
            #    for namelist in ['IONS', 'CELL']:
            #        params['parameters'].pop(namelist,None)
            #    # delete use_all_frac, if requested
            #    if final_scf_remove_use_all_frac:
            #        if 'SYSTEM' in params['parameters']:
            #            params['parameters']['SYSTEM'].pop('use_all_frac',None)
            #    # create and launch subworkflow
            #    wf_pw = PwrestartWorkflow(params=params)
            #    wf_pw.store()
            #    self.append_to_report("Launching final PW scf computation "
            #                          "(wf pk: {}) with volume={}".format(wf_pw.pk,
            #                                params['structure'].get_cell_volume()))
            #    self.attach_workflow(wf_pw)
            #    wf_pw.start()
            
            self.next(self.results_pw)
        
    @Workflow.step   
    def results_pw(self):
        main_params = self.get_parameters()
        # Retrieve the PW calculation
        # Note: order by ctime is safe if the workflows has been run on the same machine, 
        # since the loop over the run_pw step are strictly sequential 
        wf_pw_list = list(self.get_step(self.run_pw).get_sub_workflows().order_by('ctime'))
        try:
            pw_calculation = wf_pw_list[-1].get_result('pw_calculation')
            self.add_result("pw_calculation", pw_calculation)
        except ValueError:
            raise InternalError('Sub-wf {} failed with error {} '.format(
                                 wf_pw_list[-1].pk,wf_pw_list[-1].get_report()[-2]))
        
        # if requested, zip and save density files (charge, spin) in some directory 
        density_dir = main_params.get('input',{}).get('directory_for_density_files',None)
        if density_dir is not None:
            parameters = ParameterData(dict={'destination_directory': density_dir})
            _, result_dict = copy_pw_density_files_inline(parameters=parameters,
                                                          remote_folder=pw_calculation.out.remote_folder)
            self.append_to_report("Density files copied into {} (RemoteData pk: "
                                  "{})".format(density_dir,
                                              result_dict['density_remote_folder'].pk))
        
        # if the calculation has modified the structure, store it in output
        if 'relax' in main_params.get('input',{}).get('relaxation_scheme',self._default_calc):
            try:
                self.add_result("structure", pw_calculation.out.output_structure)
            except AttributeError:
                self.add_result("structure", pw_calculation.inp.structure)
        
        group_name = main_params.get('group_name',None)
        if group_name is not None:
            # create or get the group
            group, created = Group.get_or_create(name=group_name)
            if created:
                self.append_to_report("Created group '{}'".format(group_name))
            # put the pw calculation into the group
            group.add_nodes(pw_calculation)
            self.append_to_report("Adding pw calculation to group '{}'".format(group_name))

        self.append_to_report("PW workflow completed")
        
        if any( [ _.startswith('band') for _ in main_params.keys() ] ):
            self.next(self.run_bands)
        else:
            # clean scratch leftovers, if requested
            if main_params.get('input',{}).get('clean_workdir',self._clean_workdir):
                self.append_to_report("Cleaning scratch directories")
                save_calcs = [ self.get_result('pw_calculation') ]
                helpers.wipe_all_scratch(self, save_calcs)
            # then exit
            self.next(self.exit)

    @Workflow.step
    def run_bands(self):
        """
        Runs PWscf on a kpoints path
        """
        main_params = self.get_parameters()
        
        try:
            pw_calc = self.get_result('pw_calculation')
        except ValueError:
            pw_calc = main_params['pw_calculation']
        
        if not isinstance(pw_calc, PwCalculation):
            raise ValidationError("The pw_calculation passed is not an instance"
                                  " of PwCalculation")
        
        # take out parameters needed for the PW bands computation
        band_input = main_params.get('band_input',{})
        
        # get structure
        try:
            structure = pw_calc.out.output_structure
        except AttributeError:
            structure = pw_calc.inp.structure
        
        # prepare kpoints
        try:
            kpoints = main_params['band_kpoints']
        except KeyError:
            if 'distance_kpoints_in_mesh' in band_input:
                if 'distance_kpoints_in_dispersion' in band_input:
                    raise ValueError("Cannot specify both 'distance_kpoints_in_mesh'"
                        " and 'distance_kpoints_in_dispersion' for bands calculation")
                kpoints_params = ParameterData(dict={
                    'distance_kpoints_in_mesh': band_input.get(
                                        'distance_kpoints_in_mesh',
                                        self._default_distance_kpoints_in_mesh),
                    'offset': band_input.get('offset_kpoints_mesh',
                                        self._default_offset_kpoints_mesh),
                    'force_parity': band_input.get('force_parity_kpoints_mesh',
                                        self._default_force_parity_kpoints_mesh)})
            else:
                kpoints_params = ParameterData(dict={
                    'distance_kpoints_in_dispersion': band_input.get("distance_kpoints_in_dispersion",
                                               self._default_distance_kpoints_in_dispersion),
                    'kpoints_path': band_input.get("kpoints_path",None),
                    'epsilon_length': band_input.get("threshold_length_for_Bravais_lat",
                                                   self._default_epsilon_length),
                    'epsilon_angle': band_input.get("threshold_angle_for_Bravais_lat",
                                                   self._default_epsilon_angle)})
            
            res_dict = build_kpoints_inline(structure=structure,
                                            parameters=kpoints_params,
                                            store=True)
            kpoints = res_dict['output_kpoints']

        input_params = pw_calc.inp.parameters
        band_params = {'pw_parent': pw_calc,
                       'parameters': input_params,
                       'kpoints': kpoints,
                       'input': {'relaxation_scheme': 'bands'},
                       'settings': main_params.get('band_settings',{}),
                       'calculation_set': main_params.get('band_calculation_set',{})}
        
        if 'automatic_parallelization' in band_input:
            band_params['input']['automatic_parallelization'] = \
                                band_input['automatic_parallelization']

        # Now modify some of the parameters (nbnd and max_seconds)
        update_input_dict = {'pw_output_parameters': pw_calc.out.output_parameters,
                             'previous_parameters': band_params['parameters'],
                             'store': isinstance(band_params['parameters'],ParameterData)}
        if 'number_of_bands' in band_input:
            update_input_dict['settings'] = ParameterData(dict={
                'number_of_bands': band_input['number_of_bands']})
        elif 'number_of_bands_nelec_units' in band_input:
            update_input_dict['settings'] = ParameterData(dict={
                'number_of_bands_nelec_units': band_input['number_of_bands_nelec_units']})
        
        res_dict = update_params_for_bandstructure_inline(**update_input_dict)
        band_params['parameters'] = res_dict['updated_parameters']

        #Commented blocks to be deleted after testing
        
        #no_symmetries = band_input.get("no_symmetries",None)
        #if no_symmetries is True:
        #    try:
        #        band_params['parameters']['SYSTEM']['nosym'] = True
        #        band_params['parameters']['SYSTEM']['noinv'] = True
        #    except KeyError:
        #        band_params['parameters']['SYSTEM'] = {'nosym': True}
        #        band_params['parameters']['SYSTEM'] = {'noinv': True}
            
        #
        #try:
        #    band_params['parameters']['ELECTRONS']['diagonalization'] = band_input['diagonalization']
        #except KeyError:
        #    pass        

        #try:
        #    band_params['parameters']['ELECTRONS']['conv_thr'] = band_input['conv_thr']
        #except KeyError:
        #    pass        
        if (('diagonalization' in band_input) or ('conv_thr' in band_input)):
            self.append_to_report('WARNING! diagonalization or conv_thr ignored for band calculation,'
                                  'specify them in band_parameters_update.')
        
        try:
            update_dict=main_params['band_parameters_update']
            band_params['parameters'] = helpers.update_nested_dict(band_params['parameters'],update_dict)     
            if (isinstance(band_params['parameters'],ParameterData)
                and band_params['parameters']._to_be_stored):
                band_params['parameters'] = band_params['parameters'].get_dict()
        except (KeyError,AttributeError):
            pass
        
        # USELESS (done in update_params_for_bandstructure_inline and 
        # in helpers.get_pwparameterdata_from_wfparams)
        #if 'max_wallclock_seconds' in band_params['calculation_set']:
        #    max_sec = int(main_params["band_calculation_set"]["max_wallclock_seconds"]*0.97)
        #    # 3% time less to avoid  the scheduler kills pw before it safely stops
        #    band_params['parameters']['CONTROL']['max_seconds'] = max_sec
        #else:
        #    band_params['parameters']['CONTROL'].pop('max_seconds',None)
        
        wf_bands = PwbandsrestartWorkflow(params=band_params)
        wf_bands.store()
        self.append_to_report("Launching PW bands restart wf (pk: {})"
                              "".format(wf_bands.pk))
        self.attach_workflow(wf_bands)
        wf_bands.start()

        self.next(self.final_step)
    
    @Workflow.step
    def final_step(self):
        """
        Append bands to results, and compute band-gap
        """
        main_params = self.get_parameters()
        
        wf_pw_list = list(self.get_step(self.run_bands).get_sub_workflows().order_by('ctime'))
        try:
            band_calc = wf_pw_list[-1].get_result('pw_calculation')
        except ValueError:
            raise InternalError('Sub-wf {} failed with error {} '.format(
                                 wf_pw_list[-1].pk,wf_pw_list[-1].get_report()[-2]))
        
        if band_calc.has_finished_ok():
            
            self.append_to_report("PW bands calculation completed")
            bandsdata = band_calc.out.output_band

            self.append_to_report( "Band structure calculated (bandsdata "
                                   "pk: {})".format(bandsdata.pk) )
            self.add_result('band_structure', bandsdata)
            bandsdata.label = 'Electronic bands'
            bandsdata.description = ("Electronic band structure calculated with"
                                 " the workflow {}".format(self.pk))

            #compute band-gap
            _, result_dict = helpers.get_bandgap_inline(parameters=band_calc.out.output_parameters,
                                                        bands=bandsdata)
            for k,v in result_dict['output_parameters'].get_dict().iteritems():
                self.add_result(k, v)
            
            #bandsdatas = band_calc.get_outputs(BandsData, also_labels=True)
            #for i,bd in enumerate(sorted(bandsdatas)):
            #    self.append_to_report( "Band structure calculated (bandsdata "
            #                           "pk: {})".format(bd[1].pk) )
            #    result_label = ( 'band_structure' if len(bandsdatas)==1
            #                     else 'band_structure{}'.format(i+1) )
            #    self.add_result(result_label, bd[1])
            #    bd[1].label = 'Electronic bands'
            #    bd[1].description = ("Electronic band structure calculated with"
            #                                    " the workflow {}".format(self.pk))
        
            group_name = main_params.get('band_group_name',None)
            if group_name is not None:
                # create or get the group
                group, created = Group.get_or_create(name=group_name)
                if created:
                    self.append_to_report("Created group '{}'".format(group_name))
                # put the bands data into the group
                group.add_nodes(bandsdata)
                self.append_to_report("Adding bands to group '{}'".format(group_name))
      
        else:
            band_calc_output = band_calc.out.output_parameters.get_dict()
            self.append_to_report("ERROR: bands calculation (pk: {0}) failed"
                                  "\n list of warnings:\n {1}"
                                  "\n list of parser warnings:\n {2}"
                                  "".format(band_calc.pk,
                                            band_calc_output['warnings'],
                                            band_calc_output['parser_warnings']))
            raise Exception("ERROR: bands calculation failed")
              
        # clean scratch leftovers, if requested
        if main_params.get('band_input',{}).get('clean_workdir',self._clean_workdir):
            self.append_to_report("Cleaning scratch directories")
            try:
                save_calcs = [ self.get_result('pw_calculation') ]
            except (NameError, ValueError):
                save_calcs = []
            helpers.wipe_all_scratch(self, save_calcs)
        
        self.next(self.exit)
        

class PwrestartWorkflow(Workflow):
    """
    Subworkflow to handle a single QE pw.x run with a restart management in 
    case the wall time is exceeded, scf cycle did not reach convergence,
    or other kinds of failures.
    
    To be called in conjunction with the PwWorkflow (no input check!).
    """
    _max_restarts = 5
    _clean_workdir = False
    _factor_changing_degauss = 0.1
    _threshold_scf_for_changing_degauss = 30
    _threshold_abs_mag_for_saving_mag_moments = 0.001
    
    def __init__(self,**kwargs):
        super(PwrestartWorkflow, self).__init__(**kwargs)
        
    def raise_error(self,message,**kwargs):
        """
        Raise an error, and add all kwargs as attributes
        """
        self.add_attributes(kwargs)
        raise PwrestartWorkflowError(message,**kwargs)
    
    def add_resulting_structure_and_append_to_report(self,pw_calc):
        """
        Add a structure to the results, when available
        """
        try:
            self.add_result('structure',pw_calc.out.output_structure)
            self.append_to_report("We still output the last structure"
                                  " computed (pk: {})".format(
                                  self.get_result('structure').pk))
        except AttributeError:
            if 'last_clean_calc' in self.get_attributes():
                try:
                    self.add_result('structure',
                                    self.get_attribute('last_clean_calc'
                                        ).out.output_structure)
                    self.append_to_report("We still output the last "
                                          "clean structure computed "
                                          "(pk: {})".format(
                                  self.get_result('structure').pk))
                except AttributeError:
                    pass
        return
            
    @Workflow.step
    def start(self):
        """
        PW initialization.
        """
        if not self.get_step_calculations(self.start):
            self.append_to_report("Starting PW restart workflow")
        
        params = self.get_parameters()
        has_to_launch = False
        if (params.get('input',{}).get('automatic_parallelization',{}) or
            ('charge_density_folder' in params and 'retrieved_folder' not in params)):
            # runs a fake pw computation (stopped right away), either to obtain various
            # parameters (num. k-pts, fft grids, etc.) and deduce the optimal
            # parallelization settings, or to get some needed files
            # to restart from a charge density.
            # Will restart once in case of SUBMISSION_HAS_FAILED state, or
            # if no output is produced.
            pw_init_calcs = list(self.get_step_calculations(self.start).order_by('ctime'))
            if len(pw_init_calcs) == 0:
                has_to_launch = True
            else:
                if (pw_init_calcs[-1].get_state() == calc_states.SUBMISSIONFAILED
                    or 'output_parameters' not in pw_init_calcs[-1].get_outputs_dict()
                    or 'number_of_k_points' not in pw_init_calcs[-1].out.output_parameters.get_dict()):
                    if len(pw_init_calcs) >= 2:
                        # it was already restarted: stop
                        self.append_to_report("ERROR: pw.x initialization "
                                              "(pk: {0}) failed "
                                              "unexpectedly a second time".format(pw_init_calcs[-1].pk))
                        self.raise_error("ERROR: initialization failed twice",
                                         error_type='init')
                    else:
                        # try to re-submit once
                        self.append_to_report("WARNING: pw.x initialization "
                                              "(pk: {}) failed "
                                              "unexpectedly, will try to restart"
                                              " once again".format(pw_init_calcs[-1].pk))
                        has_to_launch = True
        
        if has_to_launch:
            # Take out npools from the cmdline
            if params.get('settings',{}).get('cmdline',[]):
                params['settings']['cmdline'] = helpers.take_out_npools_from_cmdline(
                                                    params['settings']['cmdline'])
            # Build calculation
            remote_folder = params.get('remote_folder',None)
            pw_calc = helpers.get_pw_calculation(params, only_initialization=True,
                                                 parent_remote_folder=remote_folder)
            self.append_to_report("Launching pw.x initialization (pk: {})"
                                  "".format(pw_calc.pk))
            self.attach_calculation(pw_calc)
            self.next(self.start)
        else:
            self.next(self.run_pw_restart)
   
        
    @Workflow.step
    def run_pw_restart(self):
        # launch PWscf code, or restart it if maximum wall time was exceeded,
        # or if pw did not reach the normal end of execution.
        # go to final step when computation succeeded in previous step.
        
        # retrieve PW parameters
        params = self.get_parameters()
        
        # Retrieve the list of pw calculations already done in this step
        pw_calc_list = list(self.get_step_calculations(self.run_pw_restart).order_by('ctime'))
            
        # retrieve attributes
        attr_dict = self.get_attributes()
        # check if previous calculation has failed unexpectedly (not due to time
        # limit nor with the parser warning 'QE pw run did not reach the end of 
        # the execution.') when has_failed or submission_has_failed is True, we 
        # try to relaunch again ONLY ONCE
        has_failed = attr_dict.get('has_failed',False)
        submission_has_failed = attr_dict.get('submission_has_failed',False)
        # parameters to update, in case they need to be changed after this step
        #update_params = helpers.default_nested_dict()
        # (attr_dict contains a dictionary looking like
        # {'update_parameters': ..., 'update_calculation_set': ...}
        # and we just take away all the 'update_' in the keys)
        update_params = {k.replace('update_',''): v for k,v in attr_dict.items()
                         if k.startswith('update_')}
        
        has_finished = False
        will_update_nbnd = False
        
        if (not pw_calc_list and params.get('input',{}).get('automatic_parallelization',{})):
            # get the optimal number of machines, of pools, of procs per machines
            # and guess the wall time needed
            pw_init_calc = list(self.get_step_calculations(self.start).order_by('ctime'))[-1]
            num_machines,num_mpiprocs_per_machine,npools,estimated_time =\
                get_parallelization_parameters(pw_init_calc,
                                               params['input']['automatic_parallelization']['max_num_machines'],
                                               params['input']['automatic_parallelization']['target_time_seconds'],
                                               calculation=params['input']['relaxation_scheme'])

            cmdline = params.get('settings',{}).get('cmdline',[])
            # take out npools if it was already there
            the_cmdline =  helpers.take_out_npools_from_cmdline(cmdline)
            the_cmdline.extend(['-nk',str(npools)])
            max_seconds = min(ceil(estimated_time/1800.)*1800, # time is set to a multiple of 30min
                params['input']['automatic_parallelization']['max_wall_time_seconds'])
            update_params = helpers.update_nested_dict(update_params,
                {'settings': {'cmdline': the_cmdline},
                 'calculation_set': {'resources': {'num_machines': num_machines,
                                                   'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                                                   },
                                     'max_wallclock_seconds': max_seconds,
                                     }
                 })
             
            self.append_to_report("Setting the number of pools to {}, number of "
                                  "machines to {}, number of procs per machine to"
                                  " {} and max_seconds to {}"
                                  "".format(npools,num_machines,num_mpiprocs_per_machine,
                                  update_params['calculation_set']['max_wallclock_seconds']))
            self.append_to_report(" -> estimated time for the calculation: {} s"
                                  "".format(estimated_time))
            
        if pw_calc_list: # if there is at least one previous calculation
            # Analyse what happened with the previous calculation
            
            # Retrieve the last pw calculation done inside this subworkflow
            pw_calc = pw_calc_list[-1]
            
            # test if it needs to be restarted
            
            # computation succeeded -> go to final step
            if pw_calc.has_finished_ok():
                has_finished = True            
            
            # case when submission failed, probably due to an IO error
            # -> try to restart once
            pw_calc_state = pw_calc.get_state()
            if pw_calc_state == calc_states.SUBMISSIONFAILED:
                if submission_has_failed:
                    # it was already restarted: stop
                    self.append_to_report("ERROR: pw.x (pk: {0}) submission failed "
                                          "unexpectedly a second time".format(pw_calc.pk))
                    self.raise_error("ERROR: submission failed twice",
                                     error_type='sub')
                else:
                    # try to re-submit once
                    self.append_to_report("WARNING: pw.x (pk: {}) submission failed "
                                          "unexpectedly, will try to restart"
                                          " once again".format(pw_calc.pk))
                    # set submission_has_failed attribute
                    self.add_attribute('submission_has_failed',True)
            # reinitialize the submission_has_failed attribute
            else:
                self.add_attribute('submission_has_failed',False)
            
            
            # Error states, in which we don't know what to do
            if (pw_calc_state not in [calc_states.FINISHED, 
                                       calc_states.FAILED, 
                                       calc_states.SUBMISSIONFAILED]):   
                # any other case leads to stop on error message
                self.append_to_report("ERROR: unexpected state ({0}) of pw.x "
                                      "(pk: {1}) calculation, stopping"
                                      "".format(pw_calc_state,pw_calc.pk))
                #has_finished = True
                self.raise_error("ERROR: unexpected state",
                                 error_type='unexpected_state')
        
            if pw_calc_state != calc_states.SUBMISSIONFAILED:
                try:
                    pw_calc_output = pw_calc.out.output_parameters
                    pw_calc_output_dict = pw_calc_output.get_dict()
                    warnings = pw_calc_output_dict['warnings']
                    parser_warnings = pw_calc_output_dict['parser_warnings']
                except (NotExistent, AttributeError, KeyError):
                    # weird case in which output parameters do not exist (most
                    # probably because output files were not even written)
                    if has_failed:
                        # this is already the second time the same calculation
                        # fails -> workflow stops
                        self.append_to_report("ERROR: pw.x (pk: {0}) did not provide"
                                              " any output a second time-> "
                                              "stopping".format(pw_calc.pk))
                        self.raise_error("ERROR: the same calculation "
                                         "provided no output twice",error_type='no_output')
                    else:
                        # first time this happens -> try to restart
                        self.append_to_report("WARNING: pw.x (pk: {}) did not provide"
                                              " any output, will try to restart"
                                              " once again".format(pw_calc.pk))
                        # set has_failed attribute
                        self.add_attribute('has_failed',True)
                else:
                    # we can continue processing the output
                    if (pw_calc_state in [calc_states.FINISHED, calc_states.FAILED]
                        and pw_calc.inp.parameters.dict['SYSTEM'].get('degauss',
                                                    _default_degauss_from_QE)>0.
                        and pw_calc.inp.parameters.dict['SYSTEM'].get('occupations',
                                                    _default_occupations_from_QE)=='smearing'
                        and 'Maximum CPU time exceeded' not in warnings):
                        # check if highest band occupation number, for each k-point
                        # and spin, is too high. If yes, we increase the number 
                        # of bands to be computed.
                        res_dict = pw_calc.get_outputs_dict()
                        if 'output_band' not in res_dict:
                            _, res_dict = get_bands_and_occupations_inline(remote_folder=pw_calc.out.remote_folder,
                                                      pw_output_parameters=pw_calc.out.output_parameters)
                        try:
                            bandsdata = res_dict['output_band']
                            # we store as an inline only if params['parameters']
                            # is a ParameterData object (otherwise anyway 
                            # provenance cannot be fully tracked)
                            update_nbands_input_dict = {'pw_output_band': bandsdata,
                                                        'store': isinstance(params['parameters'],ParameterData)}
                            if 'parameters' in update_params and update_params['parameters']:
                                update_nbands_input_dict['previous_parameters'] = \
                                                        update_params['parameters']
                            if 'max_upper_occupation_allowed' in params.get('input',{}):
                                update_nbands_input_dict['settings'] = \
                                    ParameterData(dict={'max_upper_occupation_allowed': 
                                            params['input']['max_upper_occupation_allowed']})
                            
                            res_dict = update_nbands_inline(**update_nbands_input_dict)
                            if res_dict['updated_parameters']:
                                update_params['parameters'] = res_dict['updated_parameters']
                                has_finished = False
                                will_update_nbnd = True
                                self.append_to_report("WARNING: pw.x (pk: {}) upper bands"
                                                      " have too high occupation (max {})"
                                                      " -> will try again, increasing number"
                                                      " of bands to {}".format(pw_calc.pk,
                                    res_dict['output_parameters'].get_dict()['max_of_highest_band_occupations'],
                                    res_dict['updated_parameters'].get_dict()['SYSTEM']['nbnd']))
                        except (KeyError,IndexError) as e:
                            # cannot parse the bands and occupations
                            self.append_to_report("WARNING: we cannot parse the "
                                                  "bands and occupations (error: {})"
                                                  " from pw.x (pk: {})"
                                                  " -> will try again".format(
                                                  e.message,pw_calc.pk))
                            has_finished = False
                        except TypeError:
                            # case when inline calc is prevented on purpose because
                            # no update is needed
                            pass
                        
                    if pw_calc_state == calc_states.FAILED:
                        pw_calc_input = pw_calc.inp.parameters.get_dict()
                        # TODO: implement case with 'lone vector' in warnings
                        # -> possible reason is a too high number of CPUs/not enough pools
                        # NOTE: the order is important in the following 'if'.
                        # In particular, the 3 cases with resp. the out-of-memory pb,
                        # 'QE pw run did not reach the end of the execution.' and 
                        # 'Maximum CPU time exceeded' should be just before the final
                        # 'else', and in this order.
                        # generate base inputs of inline calc. to update parameters
                        update_input_dict = {'pw_output_parameters': pw_calc_output,
                                             'store': isinstance(params['parameters'],ParameterData)}
                        if 'parameters' in update_params and update_params['parameters']:
                            update_input_dict['previous_parameters'] = \
                                                    update_params['parameters']

                        if ("The scf cycle did not reach convergence." in warnings
                            and not will_update_nbnd):
                            # case of a too slow scf convergence
                            try:
                                scf_iterations = pw_calc.get_outputs_dict().get(
                                    'output_trajectory',pw_calc.get_outputs_dict().get(
                                    'output_array',None)).get_array('scf_iterations')
                            except (KeyError,AttributeError):
                                flag_Fermi_energy_pb = False
                            else:
                                flag_Fermi_energy_pb = ((pw_calc_input['SYSTEM'].get(
                                    'smearing','') in ['methfessel-paxton','m-p','mp',
                                    'marzari-vanderbilt','cold','m-v','mv'])
                                    and (len(scf_iterations)>1)
                                    and ('degauss' in pw_calc_input['SYSTEM'])
                                    and (scf_iterations[-1]-np.average(scf_iterations[:-1]))
                                    >self._threshold_scf_for_changing_degauss)
                            
                            if flag_Fermi_energy_pb:
                                # check first if the last scf loop was suddenly much 
                                # longer -> indicates maybe a Fermi energy flip 
                                # problem for insulators with too large m-v or m-p smearing
                                # -> we try to decrease the smearing parameter
                                new_degauss = pw_calc_input['SYSTEM']['degauss']*\
                                        self._factor_changing_degauss
                                update_input_dict['settings'] = ParameterData(dict={
                                                        'degauss':  new_degauss})
                                res_dict = update_degauss_inline(**update_input_dict)
                                update_params['parameters'] = res_dict['updated_parameters']
                                self.append_to_report("WARNING: pw.x (pk: {}) failed, possibly"
                                                      " because of Fermi energy flip pb -> will try with "
                                                      "degauss={}".format(pw_calc.pk,new_degauss))
                                self.add_attribute('changed_degauss',True)
                            else:
                                # try again with different mixing scheme
                                # change mixing beta in updated parameters                    
                                res_dict = update_mixing_inline(**update_input_dict)
                                update_params['parameters'] = res_dict['updated_parameters']
                                new_mixing_beta = res_dict['updated_parameters'].get_dict()['ELECTRONS']['mixing_beta']
                                self.append_to_report("WARNING: pw.x (pk: {}) scf convergence"
                                                      " too slow -> will try with mixing "
                                #                      "scheme={}".format(pw_calc.pk,mixing_scheme))
                                                      "beta={}".format(pw_calc.pk,new_mixing_beta))
                            # pw_calc is the new last clean calculation
                            #self.add_attribute('last_clean_calc',pw_calc) 
                            # previous line commented: it's not a good idea to restart from 
                            # this calculation, since it is essentially wrong
                            # reinitialize has_failed attribute (here it is not a 
                            # "real" failure)
                            self.add_attribute('has_failed',False)

                        elif ( ( pw_calc_input['CONTROL']['calculation']=='vc-relax'
                               and pw_calc_input['SYSTEM'].get('nspin',
                                                _default_nspin_from_QE) != 1 )
                               and any(["can't open a connected unit" in _ for _ in warnings]) ):
                            # TODO: test also that the message:
                            #     lsda relaxation :  a final configuration with zero
                            #    absolute magnetization has been found
                            # is in the output file (needs to be parsed)

                            # case of known QE bug for vc-relax + magnetization (see 
                            # http://www.qe-forge.org/gf/project/q-e/tracker/?action=TrackerItemEdit&tracker_item_id=114)
                            # -> can do a vc-relax without mag. then will restart with mag.
                            res_dict = update_nspin_inline(**update_input_dict)
                            update_params['parameters'] = res_dict['updated_parameters']
                            self.append_to_report("WARNING: vc-relax + magnetization bug "
                                                  "in pw.x (pk: {}) -> will do a vc-relax "
                                                  "without spin polarization"
                                                  "".format(pw_calc.pk))
                            # Note: pw_calc is not a clean calculation, so we do
                            # not update the 'last_clean_calc' attribute
                            self.add_attribute('changed_nspin',True)
                                                
                        elif ( pw_calc_input['CONTROL']['calculation']=='vc-relax'
                               and (any(["dimensions do not match" in _ for _ in warnings])
                                 or any(["g-vectors missing !" in _ for _ in warnings]) 
                                 or any(["too many g-vectors" in _ for _ in warnings]) ) ):

                            # case of another QE bug for vc-relax + magnetization (?)
                            # (it fails at the very final scf because starting mag. 
                            # is reset at the initial non-zero value)
                            # -> output the structure and stop the workflow
                            self.append_to_report("ERROR: vc-relax + magnetization bug "
                                                  "(?), due to the change of volume and "
                                                  "FFT grid for the final scf computation "
                                                  "in pw.x calc (pk: {}) -> we stop here the restart"
                                                  " workflow (you should launch another one "
                                                  "with the updated structure)".format(pw_calc.pk))
                            # we add the structure to the results before stopping
                            try:
                                self.add_result('structure',pw_calc.out.output_structure)
                            except AttributeError:
                                pass
                            self.raise_error("ERROR: QE vc-relax failed in the final "
                                             "scf (G-vectors pb), but output structure is usable",
                                             error_type='vc_Gvectors')
                            
                        elif any(["smearing is needed" in _ for _ in warnings]):

                            # case of a metal treated without smearing
                            self.append_to_report("ERROR: the pw.x calc. (pk: {})"
                                                  " requires some smearing; "
                                                  "list of warnings:\n {}"
                                                  "".format(pw_calc.pk,warnings))
                            self.raise_error("ERROR: QE needs some smearing",
                                             error_type='smearing')

                        elif ( pw_calc_input['CONTROL']['calculation']=='vc-relax'
                               and (any(["charge is wrong" in _ for _ in warnings]) ) ):

                            # most probably, case of another QE bug for vc-relax (?)
                            # (it fails if the very final scf has to restart 
                            # because the input structure does not correspond anymore
                            # to the one studied - again G-vectors problem)
                            # -> output the structure and stop the workflow
                            self.append_to_report("ERROR: vc-relax bug (?), due to "
                                                  "the change of structure for the "
                                                  "final scf computation in pw.x calc "
                                                  "(pk: {}) -> we stop here the restart"
                                                  " workflow (you should launch another one "
                                                  "with the updated structure)".format(pw_calc.pk))
                            # we add the structure to the results before stopping
                            try:
                                self.add_result('structure',pw_calc.out.output_structure)
                            except AttributeError:
                                pass
                            self.raise_error("ERROR: QE vc-relax failed in the final "
                                             "scf (charge is wrong), but output structure is usable",
                                             error_type='vc_charge')
                            
                        elif ( pw_calc_input['CONTROL']['calculation'].startswith('vc')
                               and any(["Not enough space allocated for radial FFT" 
                                        in _ for _ in warnings])):

                            # this happens when the cell changes too much (G-vectors
                            # become too many for the initially allocated arrays - see
                            # http://www.quantum-espresso.org/faq/frequent-errors-during-execution/#5.8)
                            # -> output the structure and stop the workflow
                            self.append_to_report("ERROR: the cell changed too much "
                                                  "during the execution of the pw.x "
                                                  "calc (pk: {}) -> we stop here the restart"
                                                  " workflow (you should launch another one "
                                                  "with the updated structure)".format(pw_calc.pk))
                            # we add the structure to the results before stopping
                            try:
                                self.add_result('structure',pw_calc.out.output_structure)
                            except AttributeError:
                                pass
                            self.raise_error("ERROR: QE {} failed because cell changed "
                                             "too much, but output structure is usable".format(
                                             pw_calc_input['CONTROL']['calculation']),
                                             error_type='vc_cell')
                            
                        elif any(["read_namelists" in w for w in warnings]):   
                            # case when input parameters are incorrect
                            self.append_to_report("ERROR: incorrect input file for pw.x "
                                                  "calculation (pk: {0}) , stopping; "
                                                  "list of warnings:\n {1}".format(
                                                        pw_calc.pk,warnings))
                            self.raise_error("ERROR: incorrect input parameters",
                                             error_type='input')
                    
                        elif ( (any(["too many bands are not converged" in w for w in warnings])
                                or any(["eigenvalues not converged" in w for w in warnings]))
                                and ( pw_calc_input.get('ELECTRONS',{}).get(
                                            'diagonalization',_default_diagonalization_from_QE) 
                                            == 'david' ) ):
                            # try again from scratch with cg diagonalization 
                            res_dict = update_diag_inline(**update_input_dict)
                            update_params['parameters'] = res_dict['updated_parameters']
                            self.append_to_report("WARNING: pw.x (pk: {}) failed "
                                                  "unexpectedly, will try to restart "
                                                  "with conjugate-gradient "
                                                  "diagonalization".format(pw_calc.pk))
                            # set last_clean_calc attribute to None (to restart from
                            # scratch)
                            self.add_attribute('last_clean_calc',None)
                                                               
                        elif (any(["%%%" in w for w in warnings]) or
                              any(["Error" in w for w in warnings])):
                            # case when the code stops on any error message not 
                            # treated in the previous cases
                            if has_failed:
                                # this is already the second time the same calculation
                                # fails -> workflow stops
                                self.append_to_report("ERROR: pw.x (pk: {0}) failed "
                                                      "unexpectedly a second time-> "
                                                      "stopping; list of warnings:\n {1}"
                                                      "".format(pw_calc.pk,warnings))
                                self.add_resulting_structure_and_append_to_report(pw_calc)
                                self.raise_error("ERROR: the same calculation failed "
                                                 "twice with an error",error_type='error')
                            else:
                                # first time this happens -> try to restart
                                self.append_to_report("WARNING: pw.x (pk: {}) failed "
                                                      "with error {}, will try to "
                                                      "restart once again".format(
                                                        pw_calc.pk,warnings))
                                # set has_failed attribute
                                self.add_attribute('has_failed',True)

                        elif pw_calc.get_scheduler_error() and 'OOM' in pw_calc.get_scheduler_error():
                            # case of memory problem -> try with more nodes and/or no pools
                            # TODO: make it more general and less dependent on the scheduler exact message
                            cmdline = pw_calc.get_inputs_dict().get('settings',
                                    ParameterData(dict={})).get('cmdline',[])
                            npools = [int(p) for e,p in zip(cmdline[:-1],cmdline[1:])
                                      if (e in ('-npools','-npool','-nk'))]
                            npools = npools[0] if npools else 1
                            
                            if npools>1 or (params.get('input',{}).get('automatic_parallelization',{})
                                and pw_calc.get_resources()['num_machines'] < params['input'][
                                    'automatic_parallelization']['max_num_machines']):
                                # take out npools if it was already there (putting 
                                # npools to 1 might solve memory pb)
                                the_cmdline =  helpers.take_out_npools_from_cmdline(cmdline)
                                # increase now the number of machines
                                current_num_machines = pw_calc.get_resources()['num_machines']
                                max_num_machines = params.get('input',{}).get(
                                        'automatic_parallelization',{}).get('max_num_machines',
                                                                            current_num_machines)
                                num_machines = max([i for i in range(
                                                    current_num_machines,max_num_machines+1)
                                                    if i%current_num_machines==0])
                                if num_machines == current_num_machines:
                                    num_machines = max_num_machines
                                update_params = helpers.update_nested_dict(update_params,
                                    {'settings': {'cmdline': the_cmdline},
                                     'calculation_set': {'resources': {'num_machines': num_machines,
                                                                       },
                                                         }
                                     })
                                
                                self.append_to_report("WARNING: pw.x (pk: {}) failed "
                                                      "unexpectedly with an out-of-memory error,"
                                                      " will try to restart without pools and with "
                                                      "num_machines={}".format(pw_calc.pk,
                                                                               num_machines))
                                
                                if pw_calc_input['CONTROL'].get('wf_collect',
                                                    _default_wf_collect_from_QE):
                                    # set last_clean_calc attribute to None (to restart from
                                    # scratch because parallelization parameters changed)
                                    self.add_attribute('last_clean_calc',None)
                                
                            else:
                                self.append_to_report("ERROR: pw.x (pk: {0}) failed "
                                                      "with an out-of-memory error and "
                                                      "num_machines (resp. npools) "
                                                      "cannot be increased (resp. decreased)"
                                                      " -> stopping; scheduler error:\n {1}"
                                                      "".format(pw_calc.pk,
                                                                pw_calc.get_scheduler_error()))
                                self.raise_error("ERROR: calculation failed with an "
                                                 "out-of-memory error",error_type='oom')
                        
                        elif ( ("QE pw run did not reach the end of the execution."
                               in parser_warnings)
                               and any(["Xml data not found" in w 
                                        for w in warnings]) ):
                            # case of an unexpected stop during the initialization
                            # (probably because it hangs forever) -> try with more
                            # nodes and/or no pools, or more time
                            max_sec = pw_calc.get_max_wallclock_seconds()
                            cmdline = pw_calc.get_inputs_dict().get('settings',
                                ParameterData(dict={})).get_dict().get('cmdline',[])
                            npools = [int(p) for e,p in zip(cmdline[:-1],cmdline[1:])
                                      if (e in ('-npools','-npool','-nk'))]
                            npools = npools[0] if npools else 1
                            
                            if npools>1 or (params.get('input',{}).get('automatic_parallelization',{})
                                and pw_calc.get_resources()['num_machines'] < params['input'][
                                    'automatic_parallelization'].get('max_num_machines',1)):
                                # take out npools if it was already there (putting 
                                # npools to 1 might solve memory pb)
                                the_cmdline =  helpers.take_out_npools_from_cmdline(cmdline)
                                # now increase the number of machines
                                current_num_machines = pw_calc.get_resources()['num_machines']
                                max_num_machines = params.get('input',{}).get(
                                    'automatic_parallelization',{}).get('max_num_machines',
                                                                        current_num_machines)
                                num_machines = max([i for i in range(
                                                    current_num_machines,max_num_machines+1)
                                                    if i%current_num_machines==0])
                                if num_machines == current_num_machines:
                                    num_machines = max_num_machines
                                update_params = helpers.update_nested_dict(update_params,
                                    {'settings': {'cmdline': the_cmdline},
                                     'calculation_set': {'resources': {'num_machines': num_machines,
                                                                       },
                                                         }
                                     })
                                
                                self.append_to_report("WARNING: pw.x (pk: {}) stopped "
                                                      "during initialization -> will "
                                                      "try to restart without pools"
                                                      " and with num_machines={}"
                                                      "".format(pw_calc.pk,num_machines))

                                if pw_calc_input['CONTROL'].get('wf_collect',
                                                _default_wf_collect_from_QE):
                                    # set last_clean_calc attribute to None (to restart from
                                    # scratch because parallelization parameters changed)
                                    self.add_attribute('last_clean_calc',None)
                            
                            elif max_sec < params.get('input',{}).get(
                                    'automatic_parallelization',{}).get(
                                    'max_wall_time_seconds',0.):
                                # we increase the max wall time to the maximum available
                                the_max_sec = params['input'][
                                    'automatic_parallelization']['max_wall_time_seconds']
                                update_input_dict['settings'] = ParameterData(dict={
                                            'max_seconds': the_max_sec})
                                
                                res_dict = update_max_seconds_inline(**update_input_dict)
                                update_params['parameters'] = res_dict['updated_parameters']
                                
                                self.append_to_report("WARNING: pw.x (pk: {}) stopped "
                                                      "during initialization -> will "
                                                      "try to restart with wall time set to {}"
                                                      "".format(pw_calc.pk, int(the_max_sec)))
                                update_params = helpers.update_nested_dict(update_params,
                                    {'calculation_set': {'max_wallclock_seconds': the_max_sec}})
                                # Note: pw_calc is not a clean calculation, so we do
                                # not update the 'last_clean_calc' attribute
                            
                            elif has_failed:
                                # this is already the second time the same calculation
                                # fails -> workflow stops
                                self.append_to_report("ERROR: pw.x (pk: {0}) stopped "
                                                      "unexpectedly during "
                                                      "initialization a second time-> "
                                                      "stopping; list of warnings:\n {1}"
                                                      "".format(pw_calc.pk,warnings))
                                self.add_resulting_structure_and_append_to_report(pw_calc)
                                self.raise_error("ERROR: the same calculation stopped "
                                                 "twice",error_type='stopped')
                            else:
                                # first time this happens -> try to restart
                                self.append_to_report("WARNING: pw.x (pk: {}) stopped "
                                                      "during initialization, will "
                                                      "try to restart once again"
                                                      "".format(pw_calc.pk))
                                # set has_failed attribute
                                self.add_attribute('has_failed',True)                            
         
                        elif ("QE pw run did not reach the end of the execution." in 
                               parser_warnings):
                            # case of an unexpected stop during an scf step -> try 
                            # again from the last clean calculation, with more time
                            # if possible, otherwise trying to stop  cleanly earlier
                            # (to avoid that the scheduler kills the calc.)
                            max_sec = pw_calc.get_max_wallclock_seconds()
                            if max_sec < params.get('input',{}).get(
                                    'automatic_parallelization',{}).get(
                                    'max_wall_time_seconds',0.):
                                # we increase the max wall time to the maximum available
                                the_max_sec = params['input'][
                                    'automatic_parallelization']['max_wall_time_seconds']
                                update_input_dict['settings'] = ParameterData(dict={
                                            'max_seconds': the_max_sec})
                                
                                res_dict = update_max_seconds_inline(**update_input_dict)
                                update_params['parameters'] = res_dict['updated_parameters']
                                
                                self.append_to_report("WARNING: pw.x (pk: {}) did not reach "
                                                      "end of execution -> will "
                                                      "try to restart with wall time set to {}"
                                                      "".format(pw_calc.pk, int(the_max_sec)))
                                update_params = helpers.update_nested_dict(update_params,
                                    {'calculation_set': {'max_wallclock_seconds': the_max_sec}})
                                # Note: pw_calc is not a clean calculation, so we do
                                # not update the 'last_clean_calc' attribute
                            else:
                                # we reduce the max wall time in the pw input file, to avoid
                                # stopping in the middle of an scf step
                                res_dict = update_max_seconds_inline(**update_input_dict)
                                update_params['parameters'] = res_dict['updated_parameters']
                                self.append_to_report("WARNING: pw.x (pk: {}) did not reach "
                                                      "end of execution -> will try to restart "
                                                      "with max_seconds={}".format(pw_calc.pk,
                                                      update_params['parameters'].get_dict()['CONTROL']['max_seconds']))
                                # Note: pw_calc is not a clean calculation, so we do
                                # not update the 'last_clean_calc' attribute
                                
                        elif 'Maximum CPU time exceeded' in warnings:
                            # maximum CPU time was exceeded -> will restart
                            max_seconds = pw_calc_input['CONTROL']['max_seconds']                    
                            self.append_to_report("pw.x calculation (pk: {0}) stopped "
                                                  "because max CPU time ({1} s) was "
                                                  "reached; restarting computation "
                                                  "where it stopped"
                                                  "".format(pw_calc.pk,max_seconds))
                            # pw_calc is the new last clean calculation
                            self.add_attribute('last_clean_calc',pw_calc)
                            # reinitialize has_failed attribute (here it is not a 
                            # "real" failure)
                            self.add_attribute('has_failed',False)

                        else:
                            # case of any other kind of failure
                            if has_failed:
                                # this is already the second time the same calculation
                                # fails -> workflow stops
                                self.append_to_report("ERROR: pw.x (pk: {0}) failed "
                                                      "unexpectedly a second time-> "
                                                      "stopping; list of warnings:\n {1}"
                                                      "\n list of parser warnings:\n {2}"
                                                      "\n list of scheduler errors:\n {3}"
                                                      "".format(pw_calc.pk,
                                                                warnings,
                                                                parser_warnings,
                                                                pw_calc.get_scheduler_error()))
                                self.add_resulting_structure_and_append_to_report(pw_calc)
                                self.raise_error("ERROR: the same calculation failed "
                                                 "twice",error_type='failed')
                            else:
                                # first time this happens -> try to restart
                                self.append_to_report("WARNING: pw.x (pk: {}) failed "
                                                      "unexpectedly, will try to restart"
                                                      " once again".format(pw_calc.pk))
                                # set has_failed attribute
                                self.add_attribute('has_failed',True)
                                    

                    if will_update_nbnd:
                        # we cannot use pw_calc as the last clean calculation,
                        # because the number of bands is going to change;
                        # In that case we also reinitialize the has_failed
                        # attribute.
                        self.add_attribute('last_clean_calc',None)
                        self.add_attribute('has_failed',False)

                    if (pw_calc_state != calc_states.SUBMISSIONFAILED and
                        'atomic_magnetic_moments' in pw_calc_output_dict and
                        pw_calc_output_dict['absolute_magnetization']>=self._threshold_abs_mag_for_saving_mag_moments):
                        # re-initialize the starting magnetization so that we
                        # restart with better magnetic moments
                        update_start_mag_input_dict = {'pw_output_parameters': pw_calc_output,
                                                       'store': isinstance(params['parameters'],ParameterData)}
                        if 'parameters' in update_params and update_params['parameters']:
                            update_start_mag_input_dict['previous_parameters'] = \
                                                        update_params['parameters']
                        res_dict = update_starting_magnetization_inline(**update_start_mag_input_dict)
                        update_params['parameters'] = res_dict['updated_parameters']
                        
                    # sometimes it's more beneficial to restart from scratch,
                    # from an updated structure (in a vc-relax for instance)
                    # BAD: the FFT grid might change so the parallelization gets bad
                    # in principle... better to do it outside this sub-workflow.
                    #if (('output_structure' in pw_calc.get_outputs_dict() and
                    #    self.get_attribute('last_clean_calc') is None) or
                    #    ('output_trajectory' in pw_calc.get_outputs_dict()
                    #    and 'steps' in pw_calc.out.output_trajectory.get_arraynames()
                    #    and len(pw_calc.out.output_trajectory.get_array('steps'))>=1)):
                    #    update_structure = pw_calc.out.output_structure
                    #    self.add_attribute('last_clean_calc',None)
                    #    self.add_attribute('update_structure',update_structure)
                    #    self.append_to_report("Updating structure: {}".format(
                    #                          update_structure.pk))
                
        # deal with the update parameters
        # retrieve attributes again
        attr_dict = self.get_attributes()
        self.add_attributes({'update_{}'.format(k): 
            v if (not isinstance(v,Node) or not v._to_be_stored) else v.get_dict()
                             for k,v in update_params.items()})
        #update_structure = attr_dict.get('update_structure', None)

        # decide what to do next
        if has_finished:
            self.next(self.final_step)
        
        else:        
            # Stop if we reached the max. number of restarts (still 
            # outputting a structure if possible)
            if len(pw_calc_list) >= params.get('input',{}).get('max_restarts',
                                                     self._max_restarts):
                self.append_to_report("ERROR: Max number of restarts reached "
                                      "(last calc={})".format(pw_calc.pk))
                self.add_resulting_structure_and_append_to_report(pw_calc)
                self.raise_error("ERROR: maximum number of restarts reached "
                                 "(increase 'max_restarts')",error_type='max_restarts')

            # new parameters
            params = helpers.update_nested_dict(params,update_params)
            #if update_structure is not None:
            #    params['structure'] = update_structure
            
            if (attr_dict.get('last_clean_calc',None) is None):
                # when we (re)start from scratch, still use the initial
                # parent as last clean calculation, if it exists
                attr_dict['last_clean_calc'] = params.get('pw_parent',None)

            if attr_dict.get('last_clean_calc',None) is None:
                # Launch from scratch, or from a remote folder containing the 
                # density file(s) if it exists
                remote_folder = params.get('remote_folder',None)
                if (not remote_folder) and params.get('charge_density_folder',None):
                    # we build a remote folder from the charge density
                    # and xml data file. The latter comes from
                    # the input parameters, or if it is not provided there,
                    # from the initial calculation in the start step
                    retrieved_folder = params['retrieved_folder'] if 'retrieved_folder' in params \
                        else list(self.get_step_calculations(self.start).order_by(
                         'ctime'))[-1].out.retrieved
                    inline_params = ParameterData(dict={
                        'destination_computer_name': Code.get_from_string(
                                params["codename"]).get_computer().name,
                        })
                    
                    _, result_dict = build_pw_remote_folder_from_density_files_inline(
                        parameters = inline_params,
                        density_remote_folder = params['charge_density_folder'],
                        retrieved_folder = retrieved_folder)
                    
                    remote_folder = result_dict['pw_remote_folder']
                    if params['input']['relaxation_scheme'] not in ['nscf']:
                        store = isinstance(params['parameters'],ParameterData)
                        res_dict = update_startingpotwfc_inline(
                                previous_parameters=params['parameters'],
                                store=store)
                        if store:
                            params['parameters'] = res_dict['updated_parameters']
                        else:
                            params['parameters'] = res_dict['updated_parameters'].get_dict()
                    
                    # TODO: - fix vc-relax pb (when starting from remote
                    # folder, cell is not moving)
                    #       - optimize efficiency by doing a nscf first
                    # (otherwise the startingwfc option spoils the
                    # restart and we do as many iterations as 
                    # without initial charge density) (OBSOLET, for some reason now it's OK!) 
                        
                pw_calc = helpers.get_pw_calculation(params,
                                                     parent_remote_folder=remote_folder)
                self.append_to_report("Launching pw.x (pk: {}) from {}".format(pw_calc.pk,
                                            'remote folder (pk: {})'.format(remote_folder.pk)
                                            if remote_folder else 'scratch'))
                self.attach_calculation(pw_calc)
                # loop step on itself
                self.next(self.run_pw_restart)
                
            else:
                # Restart pw computation
                # retrieve last clean pw calculation
                last_clean_pw_calc = attr_dict['last_clean_calc']
                # prepare restarted calculation
                pw_new_calc = helpers.get_pw_calculation(params,
                                                         parent_calc=last_clean_pw_calc)
                
                # Launch restarted calculation
                self.append_to_report("Launching pw.x (pk: {}) from previous calc "
                                      "(pk: {})".format(pw_new_calc.pk,
                                                       last_clean_pw_calc.pk))
                self.attach_calculation(pw_new_calc)
                # loop step on itself
                self.next(self.run_pw_restart)
             
        
    @Workflow.step   
    def final_step(self):
        #params = self.get_parameters()
        # Retrieve the last pw calculation
        pw_calc_list = list(self.get_step_calculations(self.run_pw_restart).order_by('ctime'))
        pw_calc = pw_calc_list[-1]
        self.add_result("pw_calculation", pw_calc)
        
        self.append_to_report("PW restart workflow completed")
        
        #if params.get('clean_workdir',self._clean_workdir):
        #    self.append_to_report("Cleaning work directory")
        #    helpers.wipe_all_scratch(pw_calc_list[:-1])
            
        self.next(self.exit)


class PwbandsrestartWorkflow(Workflow):
    """
    Subworkflow to handle a single QE pw.x bands calculation, with a restart 
    management in case of failure (switching diagonalization algorithm from
    cg to david or vice-versa).
    
    To be called in conjunction with the PwWorkflow (no input check!).
    """
    _clean_workdir = False
    _max_restarts = 5
    
    def __init__(self,**kwargs):
        super(PwbandsrestartWorkflow, self).__init__(**kwargs)
            
    def raise_error(self,message,**kwargs):
        """
        Raise an error, and add all kwargs as attributes
        """
        self.add_attributes(kwargs)
        raise PwbandsrestartWorkflowError(message,**kwargs)
    
    @Workflow.step
    def start(self):
        """
        PW initialization
        """
        if not self.get_step_calculations(self.start):
            self.append_to_report("Starting PW bands restart workflow")
        params = self.get_parameters()
        has_to_launch = False
        try:
            _ = params['kpoints'].get_kpoints()
        except (KeyError,AttributeError):
            if params.get('input',{}).get('automatic_parallelization',{}):
                # runs a fake pw computation (stopped right away) to obtain the 
                # number of k-pts and deduce the 'optimal' number of pools
                # Will restart once in case of SUBMISSION_HAS_FAILED state,
                # or if there is no output file produced.
                pw_init_calcs = list(self.get_step_calculations(self.start).order_by('ctime'))
                if len(pw_init_calcs) == 0:
                    has_to_launch = True
                else:
                    if (pw_init_calcs[-1].get_state() == calc_states.SUBMISSIONFAILED
                        or 'output_parameters' not in pw_init_calcs[-1].get_outputs_dict()
                        or 'number_of_k_points' not in pw_init_calcs[-1].out.output_parameters.get_dict()):
                        if len(pw_init_calcs) >= 2:
                            # it was already restarted: stop
                            self.append_to_report("ERROR: pw.x initialization "
                                                  "(pk: {0}) failed "
                                                  "unexpectedly a second time".format(pw_init_calcs[-1].pk))
                            self.raise_error("ERROR: initialization failed twice",
                                             error_type='init')
                        else:
                            # try to re-submit once
                            self.append_to_report("WARNING: pw.x initialization "
                                                  "(pk: {}) failed "
                                                  "unexpectedly, will try to restart"
                                                  " once again".format(pw_init_calcs[-1].pk))
                            has_to_launch = True
                
        if has_to_launch:
            # Build calculation
            pw_calc = helpers.get_pw_calculation(params, only_initialization=True,
                                                 parent_calc=params['pw_parent'])
            self.append_to_report("Launching pw.x initialization (pk: {})"
                                  "".format(pw_calc.pk))
            self.attach_calculation(pw_calc)
            self.next(self.start)
        else:
            self.next(self.run_pw_bands_restart)
        
    @Workflow.step
    def run_pw_bands_restart(self):
        # launch PWscf code, or restart it if maximum wall time was exceeded,
        # or if pw did not reach the normal end of execution.
        # go to final step when computation succeeded in previous step.
        from fractions import gcd
        
        # retrieve PW parameters
        params = self.get_parameters()
        
        # Retrieve the list of pw calculations already done in this step
        pw_calc_list = list(self.get_step_calculations(self.run_pw_bands_restart).order_by('ctime'))
            
        # retrieve attributes
        attr_dict = self.get_attributes()
        # check if previous calculation has failed unexpectedly (not due to time
        # limit nor with the parser warning 'QE pw run did not reach the end of 
        # the execution.') when has_failed or submission_has_failed is True, we 
        # try to relaunch again ONLY ONCE
        has_failed = attr_dict.get('has_failed',False)
        submission_has_failed = attr_dict.get('submission_has_failed',False)
        # parameters to update, in case they need to be changed after this step
        #update_params = helpers.default_nested_dict()
        update_params = {k.replace('update_',''): v for k,v in attr_dict.items()
                         if k.startswith('update_')}
        
        has_finished = False
        
        if not pw_calc_list:
            if 'automatic_parallelization' in params.get('input',{}):
                if self.get_step_calculations(self.start):
                    # get the optimal number of machines, of pools, of procs per machines
                    # and guess the wall time needed
                    pw_init_calc = list(self.get_step_calculations(self.start).order_by('ctime'))[-1]
                    if pw_init_calc.get_state()==calc_states.FINISHED:
                        n_kpoints = pw_init_calc.out.output_parameters.get_dict()['number_of_k_points']
                    else:
                        # We do not know the number of kpoints -> set it to 1 artifically
                        # which will avoid using pools)
                        self.append_to_report("Initialization calculation (pk: {})"
                                              " failed -> the number of k-points"
                                              "is unknown and we do as if it "
                                              "is equal to 1 (no pools will be "
                                              "used)".format(pw_init_calc.pk))
                        n_kpoints = 1
                else:
                    n_kpoints = len(params['kpoints'].get_kpoints())
                if ('target_time_seconds' not in params['input']['automatic_parallelization']
                    and 'max_wall_time_seconds' in params['input']['automatic_parallelization']):
                    self.append_to_report("WARNING: you specified 'max_wall_time_seconds'"
                                          " in 'input'->'automatic_parallelization', but "
                                          " not 'target_time_seconds' -> the maximum time "
                                          " will be used as target time")
                num_machines,num_mpiprocs_per_machine,npools,estimated_time =\
                    get_parallelization_parameters(params['pw_parent'],
                                                   params['input']['automatic_parallelization']['max_num_machines'],
                                                   params['input']['automatic_parallelization'].get('target_time_seconds',
                                                        params['input']['automatic_parallelization']['max_wall_time_seconds']),
                                                   calculation=params['input']['relaxation_scheme'],
                                                   n_kpoints=n_kpoints)
    
                cmdline = params.get('settings',{}).get('cmdline',[])
                # take out npools if it was already there
                the_cmdline =  helpers.take_out_npools_from_cmdline(cmdline)
                the_cmdline.extend(['-nk',str(npools)])
                #max_seconds = params['input']['automatic_parallelization']['max_wall_time_seconds']
                max_seconds = min(ceil(estimated_time/1800.)*1800, # time is set to a multiple of 30min
                    params['input']['automatic_parallelization']['max_wall_time_seconds'])
                update_params = helpers.update_nested_dict(update_params,
                    {'settings': {'cmdline': the_cmdline},
                     'calculation_set': {'resources': {'num_machines': num_machines,
                                                       'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                                                       },
                                         'max_wallclock_seconds': max_seconds,
                                         }
                     })
                
                self.append_to_report("Setting the number of pools to {}, number of "
                                      "machines to {}, number of procs per machine to"
                                      " {} and max_seconds to {}"
                                      "".format(npools,num_machines,num_mpiprocs_per_machine,
                                      update_params['calculation_set']['max_wallclock_seconds']))
                self.append_to_report(" -> estimated time for the calculation: {} s"
                                      "".format(estimated_time))

            #if not any([_ in ('-npools','-npool','-nk') 
            #            for _ in params.get('settings',{}).get('cmdline',[])]):
            #    # get the number of k-points and find the optimal number of pools
            #    default_num_mpiprocs_per_machine = params['pw_parent'].inp.code.get_computer(
            #                                            ).get_default_mpiprocs_per_machine()
            #    num_mpiprocs_per_machine = int(params['calculation_set']['resources'].get(
            #               'num_mpiprocs_per_machine', default_num_mpiprocs_per_machine))
            #    num_procs = params['calculation_set']['resources']['num_machines']*num_mpiprocs_per_machine
            #    try:
            #        num_kpoints = len(params['kpoints'].get_kpoints())
            #    except (KeyError,AttributeError):
            #        pw_init_calc = list(self.get_step_calculations(self.start).order_by('ctime'))[-1]    
            #        try:
            #            num_kpoints = pw_init_calc.res.number_of_k_points
            #        except AttributeError:
            #            # Dummy calculation failed.
            #            # With the follwoing you get npool=1
            #            num_kpoints=2
            #            self.append_to_report("WARNING: initial pw.x (pk: {0}) failed"
            #                                  "to output number_of_k_points. \n Setting npools=1"
            #                                  "".format(pw_init_calc.pk))

                # OLD: npools was the greatest common divisor of num_procs and num_kpoints
                # Pb: this gives typically too small number of pools
                #npools = gcd(num_kpoints,num_procs)
                # NEW: npools should be a divisor of num_procs but simply
                # smaller than num_kpoints
            #    for i in range(num_kpoints/2,0,-1):
            #        if np.mod(num_procs,i)==0:
            #            npools = i
            #            break
            #    cmdline = params.get('settings',{}).get('cmdline',[])
            #    cmdline.extend(['-nk',str(npools)])
            #    self.append_to_report("Setting the number of pools to {}".format(npools))
            #    update_params['settings']['cmdline'] = cmdline
        
        else: # if there is at least one previous calculation
            # Analyse what happened with the previous calculation
            
            # Retrieve the last pw calculation done inside this subworkflow
            pw_calc = pw_calc_list[-1]
            pw_calc_state = pw_calc.get_state()
            
            # test if it needs to be restarted
            
            # computation succeeded -> go to final step
            if pw_calc.has_finished_ok():
                has_finished = True            
            
            # case when submission failed, probably due to an IO error
            # -> try to restart once
            if pw_calc_state == calc_states.SUBMISSIONFAILED:
                if submission_has_failed:
                    # it was already restarted: stop
                    self.append_to_report("ERROR: pw.x (pk: {0}) submission failed "
                                          "unexpectedly a second time".format(pw_calc.pk))
                    self.raise_error("ERROR: submission failed twice",
                                     error_type='sub')
                else:
                    # try to re-submit once
                    self.append_to_report("WARNING: pw.x (pk: {}) submission failed "
                                          "unexpectedly, will try to restart"
                                          " once again".format(pw_calc.pk))
                    # set submission_has_failed attribute
                    self.add_attribute('submission_has_failed',True)
            # reinitialize the submission_has_failed attribute
            else:
                self.add_attribute('submission_has_failed',False)
            
            
            # Error states, in which we don't know what to do
            if (pw_calc_state not in [calc_states.FINISHED, 
                                       calc_states.FAILED,
                                       calc_states.SUBMISSIONFAILED]):   
                # any other case leads to stop on error message
                self.append_to_report("ERROR: unexpected state ({0}) of pw.x "
                                      "(pk: {1}) calculation, stopping"
                                      "".format(pw_calc_state,pw_calc.pk))
                #has_finished = True
                self.raise_error("ERROR: unexpected state",
                                 error_type='unexpected_state')
        
            
            if pw_calc_state == calc_states.FAILED:
 
                pw_calc_input = pw_calc.inp.parameters.get_dict()
                try:
                    pw_calc_output = pw_calc.out.output_parameters
                    update_input_dict = {'pw_output_parameters': pw_calc_output,
                                         'store': isinstance(params['parameters'],ParameterData)}
                    pw_calc_output_dict = pw_calc_output.get_dict()
                    warnings = pw_calc_output_dict['warnings']
                    parser_warnings = pw_calc_output_dict['parser_warnings']
    
                    if any(["read_namelists" in w for w in warnings]):   
                        # any other case leads to stop on error message
                        self.append_to_report("ERROR: incorrect input file for pw.x "
                                              "calculation (pk: {0}) , stopping; "
                                              "list of warnings:\n {1}".format(
                                                    pw_calc.pk,warnings))
                        self.raise_error("ERROR: incorrect input parameters",
                                         error_type='input')
                
                    elif ((not has_failed) and 
                            ((any(["too many bands are not converged" in w for w in warnings])
                            or any(["eigenvalues not converged" in w for w in warnings]))
                            ) ):
                        # case of an unexpected stop -> switch to another 
                        # diagonalization algorithm
                        if 'parameters' in update_params and update_params['parameters']:
                            update_input_dict['previous_parameters'] = \
                                                    update_params['parameters']
                        res_dict = update_diag_inline(**update_input_dict)
                        update_params['parameters'] = res_dict['updated_parameters']
                        new_diag_algo = update_params['parameters'].dict.ELECTRONS['diagonalization']
                        self.append_to_report("WARNING: pw.x (pk: {}) did not reach "
                                              "end of execution and got the "
                                              "following messages: {} -> will try "
                                              "to restart changing diagonalization"
                                              " to {}".format(pw_calc.pk,
                                                              warnings,
                                                              new_diag_algo))
                        # set has_failed attribute
                        self.add_attribute('has_failed',True)
                        # we restart from scratch because diag. algo. changed
                        self.add_attribute('last_clean_calc',None)

                    elif (("QE pw run did not reach the end of the execution."
                               in parser_warnings)
                          and len(warnings)==0 ):
                        # case of an unexpected stop (typically during the 
                        # initialization) -> try with more
                        # nodes and/or no pools, or more time
                        max_sec = pw_calc.get_max_wallclock_seconds()
                        cmdline = pw_calc.get_inputs_dict().get('settings',
                            ParameterData(dict={})).get_dict().get('cmdline',[])
                        npools = [int(p) for e,p in zip(cmdline[:-1],cmdline[1:])
                                  if (e in ('-npools','-npool','-nk'))]
                        npools = npools[0] if npools else 1
                        
                        if (npools>1 or (params.get('input',{}).get('automatic_parallelization',{})
                            and pw_calc.get_resources()['num_machines'] < params['input'][
                                'automatic_parallelization'].get('max_num_machines',1))
                            or max_sec < params.get('input',{}).get(
                                'automatic_parallelization',{}).get(
                                'max_wall_time_seconds',0.)):
                            # take out npools if it was already there (putting 
                            # npools to 1 might solve memory pb)
                            the_cmdline =  helpers.take_out_npools_from_cmdline(cmdline)
                            the_max_sec = params['input'][
                                'automatic_parallelization']['max_wall_time_seconds']
                            # now increase the number of machines
                            current_num_machines = pw_calc.get_resources()['num_machines']
                            max_num_machines = params.get('input',{}).get(
                                'automatic_parallelization',{}).get('max_num_machines',
                                                                    current_num_machines)
                            num_machines = max([i for i in range(
                                                current_num_machines,max_num_machines+1)
                                                if i%current_num_machines==0])
                            if num_machines == current_num_machines:
                                num_machines = max_num_machines
                            # we increase also the max wall time to the maximum available
                            update_input_dict['settings'] = ParameterData(dict={
                                        'max_seconds': the_max_sec})
                            res_dict = update_max_seconds_inline(**update_input_dict)
                            update_params['parameters'] = res_dict['updated_parameters']
                            
                            update_params = helpers.update_nested_dict(update_params,
                                {'settings': {'cmdline': the_cmdline},
                                 'calculation_set': {'resources': {'num_machines': num_machines,
                                                                   },
                                                     'max_wallclock_seconds': the_max_sec,
                                                     }
                                 })
                            
                            self.append_to_report("WARNING: pw.x (pk: {}) did not "
                                                  "reach the end of the execution ->  "
                                                  "will try to restart without pools"
                                                  ", with num_machines={} and with"
                                                  " wall time set to {}"
                                                  "".format(pw_calc.pk,num_machines,
                                                            int(the_max_sec)))
                            
                            if pw_calc_input['CONTROL'].get('wf_collect',
                                            _default_wf_collect_from_QE):
                                # set last_clean_calc attribute to None (to restart from
                                # scratch because parallelization parameters changed)
                                self.add_attribute('last_clean_calc',None)
                            
                        elif has_failed:
                            # this is already the second time the same calculation
                            # fails -> workflow stops
                            self.append_to_report("ERROR: pw.x (pk: {0}) stopped  "
                                                  "unexpectedly a second time-> "
                                                  "stopping; list of warnings:\n {1}"
                                                  "".format(pw_calc.pk,warnings))
                            self.raise_error("ERROR: the same calculation stopped "
                                             "twice",error_type='stopped')
                        else:
                            # first time this happens -> try to restart
                            self.append_to_report("WARNING: pw.x (pk: {}) did not "
                                                  "reach the end of the execution, "
                                                  "will try to restart once again"
                                                  "".format(pw_calc.pk))
                            # set has_failed attribute
                            self.add_attribute('has_failed',True)                            
                                                
                    elif 'Maximum CPU time exceeded' in warnings:
                        # maximum CPU time was exceeded -> will restart
                        max_seconds = pw_calc_input['CONTROL']['max_seconds']                    
                        self.append_to_report("pw.x calculation (pk: {0}) stopped "
                                              "because max CPU time ({1} s) was "
                                              "reached; restarting computation "
                                              "where it stopped"
                                              "".format(pw_calc.pk,max_seconds))
                        # pw_calc is the new last clean calculation
                        self.add_attribute('last_clean_calc',pw_calc)
                        # reinitialize has_failed attribute (here it is not a 
                        # "real" failure)
                        self.add_attribute('has_failed',False)
                    
                    else:
                        # case of another kind of failure
                        if has_failed:
                            # this is already the second time the same calculation
                            # fails -> workflow stops
                            self.append_to_report("ERROR: pw.x (pk: {0}) failed "
                                                  "unexpectedly a second time-> "
                                                  "stopping; list of warnings:\n {1}"
                                                  "\n list of parser warnings:\n {2}"
                                                  "".format(pw_calc.pk,
                                                            warnings,
                                                            parser_warnings))
                            self.raise_error("ERROR: the same calculation failed "
                                             "twice",error_type='failed')
                        else:
                            # first time this happens -> try to restart
                            self.append_to_report("WARNING: pw.x (pk: {}) failed "
                                                  "unexpectedly, will try to restart"
                                                  " once again".format(pw_calc.pk))
                            # set has_failed attribute
                            self.add_attribute('has_failed',True)
                            
                except (NotExistent, AttributeError):
                    # weird case in which output parameters do not exist (most
                    # probably because output files where not even written)
                    if has_failed:
                        # this is already the second time the same calculation
                        # fails -> workflow stops
                        self.append_to_report("ERROR: pw.x (pk: {0}) did not provide"
                                              " any output a second time-> "
                                              "stopping".format(pw_calc.pk))
                        self.raise_error("ERROR: the same calculation "
                                         "provided no output twice",
                                         error_type='no_output')
                    else:
                        # first time this happens -> try to restart
                        self.append_to_report("WARNING: pw.x (pk: {}) did not provide"
                                              " any output, will try to restart"
                                              " once again".format(pw_calc.pk))
                        # set has_failed attribute
                        self.add_attribute('has_failed',True)
                
        # decide what to do next
        if has_finished:
            self.next(self.final_step)
        
        else:        

            # Stop if we reached the max. number of restarts (still 
            # outputting a structure if possible)
            if len(pw_calc_list) >= params.get('input',{}).get('max_restarts',
                                                     self._max_restarts):
                self.append_to_report("ERROR: Max number of restarts reached "
                                      "(last calc={})".format(pw_calc.pk))
                self.raise_error("ERROR: maximum number of restarts reached "
                                 "(increase 'max_restarts')",error_type='max_restarts')

            self.add_attributes({'update_{}'.format(k): 
                v if (not isinstance(v,Node) or not v._to_be_stored) else v.get_dict()
                                 for k,v in update_params.items()})
            # new parameters
            params = helpers.update_nested_dict(params,update_params)
            
            if ('last_clean_calc' in self.get_attributes() and 
                self.get_attribute('last_clean_calc') is not None):
                parent_calc = self.get_attribute('last_clean_calc')
            else:
                parent_calc = params['pw_parent']
            pw_new_calc = helpers.get_pw_calculation(params,
                                                     parent_calc=parent_calc)
            
            # Launch bands calculation
            self.append_to_report("Launching pw.x (pk: {}) from previous calc "
                                  "(pk: {})".format(pw_new_calc.pk,
                                                    parent_calc.pk))
            self.attach_calculation(pw_new_calc)
            # loop step on itself
            self.next(self.run_pw_bands_restart)

    @Workflow.step   
    def final_step(self):
        #params = self.get_parameters()
        # Retrieve the last pw calculation
        pw_calc_list = list(self.get_step_calculations(self.run_pw_bands_restart).order_by('ctime'))
        pw_calc = pw_calc_list[-1]
        self.add_result("pw_calculation", pw_calc)
        
        self.append_to_report("PW bands restart workflow completed")
        
        #if params.get('clean_workdir',self._clean_workdir):
        #    self.append_to_report("Cleaning work directory")
        #    helpers.wipe_all_scratch(pw_calc_list[:-1])
            
        self.next(self.exit)

