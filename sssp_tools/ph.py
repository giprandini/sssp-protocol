# -*- coding: utf-8 -*-
import os, numpy
from aiida.orm.workflow import Workflow
from aiida.orm import DataFactory, CalculationFactory, Code, Group, \
                       load_node, load_workflow
from aiida.workflows.user.epfl_theos.quantumespresso import helpers
from aiida.common.datastructures import calc_states
from aiida.common.example_helpers import test_and_get_code
from aiida.orm.calculation.inline import make_inline,optional_inline
from aiida.common.exceptions import ValidationError
from aiida.orm.calculation.inline import InlineCalculation
from datetime import datetime
   
__copyright__ = u"Copyright (c), This file is part of the AiiDA-EPFL Pro platform. For further information please visit http://www.aiida.net/. All rights reserved"
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file."
__version__ = "0.1.0"
__authors__ = "Nicolas Mounet, Andrea Cepellotti, Giovanni Pizzi, Davide Campi."

ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')
PhCalculation = CalculationFactory('quantumespresso.ph')
RemoteData = DataFactory('remote')


@make_inline
def copy_dvscf_files_inline(parameters,remote_folder):
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

    from aiida.backends.utils import get_authinfo
    #from aiida.execmanager import get_authinfo
    import os
    
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
    calcuuid = remote_folder.inp.remote_folder.uuid
    t_source = get_authinfo(computer=remote_folder.get_computer(),
                            aiidauser=remote_folder.get_user()).get_transport()
    source_dir = os.path.join(remote_folder.get_remote_path(),
                              PhCalculation._OUTPUT_SUBFOLDER)


    with t_source,t_dest:
        # get the density files name
        ph_source_dir = os.path.join(source_dir,'_ph0')
        #density_dir = os.path.join(source_dir,
        #                           "{}.save".format(PwCalculation._PREFIX))
        t_source.chdir(ph_source_dir)
        content_list = t_source.listdir()
        #density_files = [f for f in content_list 
        #                 if (any([ (f.find(_) != -1) for _ in ('dvscf',
        #                                                       'phsave',
        #                                                          )])                                                        
        #                     )]

        density_files = [f for f in content_list 
                         if (any([f.startswith(_) for _ in ('aiida.dvscf',
                                                            'aiida.phsave',
                                                            'aiida.drho',
                                                           )])                                                        
                             )]
        
        # zip the density files (keeping the original files)
        for density_file in density_files:
            #_,stdout,stderr = t_source.exec_command_wait(" ".join(
            #    ["gzip","-c", "-r", density_file,">","{}.gz".format(density_file)]))
            _,stdout,stderr = t_source.exec_command_wait(" ".join(
                ["tar", "-zcf","{}.tar.gz".format(density_file), density_file]))
            # -c option is to keep original file and output on std output
            if stderr:
                raise InternalError("Error while compressing the density "
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
        density_file_paths = [os.path.join(ph_source_dir,"{}.tar.gz".format(f)) for f in density_files ]
            
	    #"{}.save".format(PwCalculation._PREFIX),
            #if f != paw_file else os.path.join(source_dir,"{}.gz".format(f))
            #for f in density_files]

        for density_file_gz in density_file_paths:
            if t_dest._machine == t_source._machine:
                t_source.copy(density_file_gz,final_dest_dir)
            else:
                t_source.copy_from_remote_to_remote(t_dest,density_file_gz,
                                                    final_dest_dir)
            t_source.remove(density_file_gz)
    
    dvscf_remote_folder = RemoteData(computer=computer_dest,
                                       remote_path=final_dest_dir)
    
    return {'dvscf_remote_folder': dvscf_remote_folder}


@optional_inline
def distribute_qpoints_inline(structure,**kwargs):
    """
    :param retrieved: a FolderData object with the retrieved node of
            a PhCalculation
            
    :return: a dictionary of the form {'qpoints1': qpoint1, 'qpoints2': qpoint2}
        where each qpointN is a KpointsData object with a single qpoint
    """
    from aiida.parsers.plugins.quantumespresso.raw_parser_ph import parse_ph_text_output
    
    FolderData = DataFactory('folder')
    retrieved_ph = kwargs.pop("retrieved")
    
    if kwargs:
        raise ValueError("Unrecognized kwargs left")
    
    if not isinstance(retrieved_ph,FolderData):
        raise ValueError("Input should be a FolderData")
    
    dynmat_file = retrieved_ph.get_abs_path("{}{}".format(
                             PhCalculation()._OUTPUT_DYNAMICAL_MATRIX_PREFIX,0))
    with open(dynmat_file,'r') as f:
        lines = f.readlines()
    
    try:
        _ = [ float(i) for i in lines[0].split()]
    except IndexError:
        raise IndexError("File '{}' is empty (it should contain "
                         "the list of q-points)".format(dynmat_file))
    except ValueError:
        raise ValueError("File '{}' does not contain the list of "
                         "q-points".format(dynmat_file))
    
    # read also text output file to find the cell & lattice parameter
    #out_file = retrieved_ph.get_abs_path("{}".format(PhCalculation()._OUTPUT_FILE_NAME))
    #with open(out_file,'r') as f:
    #    out_lines = f.readlines()
    #out_data,_ = parse_ph_text_output(out_lines)
    #cell = out_data['cell']
    cell = structure.cell
    lattice_parameter = numpy.linalg.norm(cell[0])
    twopiovera = 2.*numpy.pi/lattice_parameter
    # read q-points, converting them from 2pi/a coordinates to angstrom-1
    qpoint_coordinates = [ [float(i)*twopiovera for i in j.split()] for j in lines[2:] ]
    
    qpoint_dict = {}
    for count,q in enumerate(qpoint_coordinates):
        qpoint = KpointsData()
        qpoint.set_cell(cell)
        qpoint.set_kpoints([q],cartesian=True)
        qpoint_dict['qpoint_{}'.format(count)] = qpoint
    
    return qpoint_dict


def get_distribute_qpoints_results(structure=None,**kwargs):
    """
    Get the results from the distribute_qpoints_inline function:
    - if there exists already an inline calculation with the same inputs,
    it does not relaunch it, it gets instead the output dictionary of the previously
    launched function,
    - otherwise, launches the distribute_qpoints_inline function
    and get its result.
    """    
    result_dict = None
    inputs_list = sorted([v.pk for k,v in kwargs.iteritems()])
    for ic in InlineCalculation.query(inputs=structure).order_by('ctime'):
        ic_inputs_list = sorted([v.pk for k,v in ic.get_inputs_dict().iteritems()
                                 if not k.startswith('structure')])
        try:
            if ( ic.get_function_name() == 'distribute_qpoints_inline'
                 and ic.inp.structure.uuid == structure.uuid
                 and ic_inputs_list==inputs_list
                 and 'qpoint_0' in ic.get_outputs_dict()):
                #if not check_also_outputs:
                result_dict = ic.get_outputs_dict()
                #else:
                #    epsilon = 1e-8
                #    from aiida.workflows.user.epfl_theos.dbimporters.utils import objects_set
                #    ic_result_dict = ic.get_outputs_dict()
                #    tmp_result_dict = distribute_qpoints_inline(
                #                   structure=structure, store=False)
                #    tmp_cells = [v.cell for k,v in tmp_result_dict.items()
                #                           if 'structure' in k]
                #    ic_cells = [v.cell for k,v in ic_result_dict.items()
                #                           if 'structure' in k and k.count('_')==3]
                #    tmp_ic_dup_cells = [objects_are_equal(objects_set(tmp_cells,epsilon=epsilon),
                #                        objects_set(tmp_cells+[c],epsilon=epsilon),epsilon=epsilon)
                #                        for c in ic_cells]
                #    ic_tmp_dup_cells = [objects_are_equal(objects_set(ic_cells,epsilon=epsilon),
                #                        objects_set(ic_cells+[c],epsilon=epsilon),epsilon=epsilon)
                #                        for c in tmp_cells]
                #    if (objects_are_equal(tmp_result_dict['output_parameters'].get_dict(),
                #        ic.out.output_parameters.get_dict()) and
                #        all(tmp_ic_dup_cells) and all(ic_tmp_dup_cells)
                #        and len(tmp_cells)==len(ic_cells)):
                #        result_dict = ic_result_dict
                        
        except AttributeError:
            pass
    
    #if result_dict is not None:
    #    print "distribute_qpoints_inline already run -> we do not re-run"
    if result_dict is None:
        #print "Launch distribute_qpoints_inline..."
        result_dict = distribute_qpoints_inline(
                    structure=structure, store=True, **kwargs)
    
    return result_dict

@make_inline
def recollect_qpoints_inline(**kwargs):
    """
    Collect dynamical matrix files into a single folder, putting a different
    number at the end of each final dynamical matrix file corresponding to
    its place in the list of q-points.
    
    :param **kwargs: all the folder data with the dynamical matrices
        (the initial calculation folder with the list of q-points should be in 
        kwargs['initial_folder'])
    :return: a dictionary of the form {'retrieved': folder}
        where folder is a FolderData with all the dynamical matrices.
    """
    # prepare an empty folder with the subdirectory of the dynamical matrices
    FolderData = DataFactory('folder')
    folder = FolderData()
    folder_path = os.path.normpath(folder.get_abs_path('.'))
    os.mkdir( os.path.join( folder_path, PhCalculation()._FOLDER_DYNAMICAL_MATRIX ) )
    
    initial_folder = kwargs.pop('initial_folder')
    
    # add the dynamical-matrix-0 file first
    file_dynmat0 = "{}{}".format(PhCalculation()._OUTPUT_DYNAMICAL_MATRIX_PREFIX,0)
    folder.add_path( initial_folder.get_abs_path(file_dynmat0), file_dynmat0)
    
    for k, src_folder in kwargs.iteritems():
        file_name = PhCalculation()._OUTPUT_DYNAMICAL_MATRIX_PREFIX
        new_file_name = "{}{}".format(PhCalculation()._OUTPUT_DYNAMICAL_MATRIX_PREFIX,
                                      int(k.split('_')[1])+1)
        folder.add_path( src_folder.get_abs_path(file_name), new_file_name )
    
    return {'retrieved': folder}


@make_inline
def recollect_allremotes_inline(parameters,**kwargs):
    """
    gne gne gne
    """
    
    from aiida.common.utils import get_new_uuid
    from aiida.backends.utils import get_authinfo
    #from aiida.execmanager import get_authinfo
    import os

    params_dict = parameters.get_dict()
    computer_dest_name = params_dict.get('destination_computer_name',None)
    if computer_dest_name:
        computer_dest = Computer.get(computer_dest_name)
    else:
        computer_dest = kwargs.itervalues().next().get_computer()
    t_dest = get_authinfo(computer=computer_dest,
                          aiidauser=kwargs.itervalues().next().get_user()).get_transport()

    t_source = get_authinfo(computer=computer_dest,
                          aiidauser=kwargs.itervalues().next().get_user()).get_transport()
    dest_dir = params_dict['destination_directory']

    #open transport
    with t_dest, t_source:
    # build the destination folder
        t_dest.chdir(dest_dir)

        folduuid=get_new_uuid()
    # we do the same logic as in the repository and in the working directory,
    # i.e. we create the final directory where to put the file splitting the
    # uuid of the calculation
        t_dest.mkdir(folduuid[:2], ignore_existing=True)
        t_dest.chdir(folduuid[:2])
        t_dest.mkdir(folduuid[2:4], ignore_existing=True)
        t_dest.chdir(folduuid[2:4])
        t_dest.mkdir(folduuid[4:])
        t_dest.chdir(folduuid[4:])

        final_dest_dir = t_dest.getcwd()

   
        for k, rem_folder in kwargs.iteritems():

            t_dest.mkdir(str(int(k.split('_')[1])+1))
            t_dest.chdir(str(int(k.split('_')[1])+1))

            final_q_dest_dir = t_dest.getcwd()
        
            
            init_folder = rem_folder.out.remote_folder.out.dvscf_remote_folder.get_remote_path()
            t_source.chdir(init_folder)
            dvscf_files = t_source.listdir()

            #NOPE dvscf_files = rem_folder.out.remote_folder.out.dvscf_remote_folder.get_folder_list()

            #dvscf_files = rem_folder.out.remote_folder.out.dvscf_remote_folder
            #density_file_gz = [os.path.join(init_folder,"*.tar.gz")]

            dvscf_file_paths = [os.path.join(init_folder,"{}".format(f)) for f in dvscf_files ]
            
            
            for dvscf_file_gz in dvscf_file_paths:
            #if t_dest._machine == t_source._machine:
                t_dest.copy(dvscf_file_gz,final_q_dest_dir)

            #t_dest.copytree(init_folder,final_q_dest_dir)

            t_dest.chdir(final_dest_dir)

    
    final_dvscf_remote_folder = RemoteData(computer=computer_dest,
                                       remote_path=final_dest_dir)
    
    return {'global_dvscf_folder': final_dvscf_remote_folder}


def get_ph_calculation(wf_params, only_initialization=False,
                       parent_calc=None, store=True):
    """
    :return ph_calc: a stored PhCalculation, ready to be submitted
    """
    # default max number of seconds for a calculation with only_initialization=True
    # (should be largely sufficient)
    default_max_seconds_only_init = 1800

    # ph.x has 5% time less than the wall time, to avoid the scheduler kills 
    # the calculation before it safely stops
    ph_parameters = wf_params["parameters"]
    if 'max_seconds' not in ph_parameters['INPUTPH']:
        max_wallclock_seconds = wf_params['calculation_set']['max_wallclock_seconds']
        ph_parameters['INPUTPH']['max_seconds'] = int(0.95*max_wallclock_seconds)
    
    if parent_calc is None:
        code = Code.get_from_string(wf_params['codename'])
        calc = code.new_calc()
        parent_calc = wf_params['pw_calculation']
        calc.use_parent_calculation(parent_calc)
    else:
        calc = parent_calc.create_restart(force_restart=True)
        ph_parameters['INPUTPH']['recover'] = True
            
    # by default, set the resources as in the parent calculation
    calc.set_resources(parent_calc.get_resources())
    # then update resources & time using calculation_set
    calc = helpers.set_the_set(calc, wf_params['calculation_set'])

    calc.use_parameters(ParameterData(dict=ph_parameters))    
    calc.use_qpoints(wf_params['qpoints'])

    if ( ('settings' in wf_params or 'settings' in parent_calc.get_inputs_dict())
         or only_initialization ):
        settings_dict = parent_calc.get_inputs_dict().get('settings',
                                                          ParameterData(dict={})).get_dict()
        
        # clean-up unwanted variables
        # TODO: probably other variables need to be cleaned up
        try:
            del settings_dict['also_bands'] # Only accepted by pw
        except KeyError:
            # It was not there, all good
            pass

        settings_dict = helpers.update_nested_dict(settings_dict, wf_params.get('settings', {}))
        if only_initialization:
            settings_dict['ONLY_INITIALIZATION'] = only_initialization
            calc.set_max_wallclock_seconds(default_max_seconds_only_init)
        calc.use_settings(ParameterData(dict=settings_dict))
    
    if store:
        calc.store_all()
    
    return calc
        

def get_ph_calc_with_parameters(wf_params, ignored_keys=['INPUTPH|alpha_mix(1)',
        'INPUTPH|niter_ph','INPUTPH|nmix_ph','INPUTPH|fildrho',
        'INPUTPH|fildvscf','INPUTPH|max_seconds'],**kwargs):
    """
    Find all ph calculations already run with the same parameters.
    :param wf_params: a dictionary with all the wf parameters
    :param ignored_keys: keys to be ignored from the parameters
    :return: the list of calculations.
    """
    from aiida.workflows.user.epfl_theos.dbimporters.utils import objects_are_equal
    from aiida.workflows.user.epfl_theos.quantumespresso.helpers import take_out_keys_from_dictionary
    eps = 1e-8
    ph_ref = get_ph_calculation(wf_params, store=False, **kwargs)
    parent_calc = kwargs.get('parent_calc',wf_params['pw_calculation'])
    
    phs = PhCalculation.query(inputs__inputs=parent_calc.dbnode,
        outputs__outputs__output_links__label__startswith='qpoint').distinct()
    the_phs = []
    for ph in [_ for _ in phs if 'retrieved' in _.get_outputs_dict()]:
        inputs_dict_ref = {k:v for k,v in ph_ref.get_inputs_dict().items()
                           if not k.startswith('code')}
        inputs_dict = {k:v for k,v in ph.get_inputs_dict().items()
                           if not k.startswith('code')}
        flag_identical = True
        for k,v in inputs_dict_ref.items():
            if isinstance(v,ParameterData):
                flag_identical = (flag_identical and k in inputs_dict 
                    and isinstance(inputs_dict[k],ParameterData))
                if flag_identical:
                    p1 = v.get_dict()
                    p2 = inputs_dict.pop(k).get_dict()
                    if k=='parameters':
                        take_out_keys_from_dictionary(p1,ignored_keys)
                        take_out_keys_from_dictionary(p2,ignored_keys)
                    flag_identical = (flag_identical and objects_are_equal(p1,p2,epsilon=eps))
                
            if isinstance(v,KpointsData):
                try:
                    kpts = v.get_kpoints()
                    flag_kpts_list = True
                except AttributeError:
                    kpts = v.get_kpoints_mesh()
                    flag_kpts_list = False
                if flag_kpts_list:
                    flag_identical = (flag_identical and k in inputs_dict
                        and isinstance(inputs_dict[k],KpointsData)
                        and 'array|kpoints' in inputs_dict[k].get_attrs()
                        and objects_are_equal(inputs_dict.pop(k).get_kpoints(),kpts,epsilon=eps))
                else:
                    flag_identical = (flag_identical and k in inputs_dict
                        and isinstance(inputs_dict[k],KpointsData)
                        and 'mesh' in inputs_dict[k].get_attrs()
                        and objects_are_equal(inputs_dict.pop(k).get_kpoints_mesh(),kpts,epsilon=eps))
             
            if isinstance(v,RemoteData):
                flag_identical = (flag_identical and k in inputs_dict
                    and isinstance(inputs_dict[k],RemoteData)
                    and inputs_dict.pop(k).pk==v.pk)
            
            if not flag_identical:
                break
        
        flag_identical = flag_identical and (not inputs_dict)
        if flag_identical:
            the_phs.append(ph)
                
    return the_phs


class PhWorkflow(Workflow):
    """
    General workflow to launch Phonon calculations
    
    Docs missing
    """
    _clean_workdir = False
    _use_qgrid_parallelization = False
    _default_QE_epsil = False
    
    def __init__(self,**kwargs):
        super(PhWorkflow, self).__init__(**kwargs)
    
    @Workflow.step
    def start(self):
        """
        Check the input parameters of the Workflow
        """
        self.append_to_report("Checking PhWorkflow input parameters")
        
        mandatory_ph_keys = [('codename',basestring,'The name of the ph.x code'),
                             ('qpoints',KpointsData,'A KpointsData object with the qpoint mesh used by PHonon'),
                             ('calculation_set',dict,'A dictionary with resources, walltime, ...'),
                             ('parameters',dict,"A dictionary with the PH input parameters"),
                             ]
        
        main_params = self.get_parameters()
        
        helpers.validate_keys(main_params, mandatory_ph_keys)
        
        # check the ph code
        test_and_get_code(main_params['codename'], 'quantumespresso.ph',
                          use_exceptions=True)
        
        # ATM only qgrid parallelization is supported
        _ = main_params.get('input',{}).get('use_qgrid_parallelization', 
                                                              self._use_qgrid_parallelization)
        
        # Avoid a frequent crash due to input
        if ( main_params['pw_calculation'].res.smearing_method and 
              main_params['parameters']['INPUTPH'].get('epsil', self._default_QE_epsil) ):
            raise ValueError("Metals require parameter['INPUTPH']['epsil'] = False")
        
        self.next(self.run_ph)
        
    @Workflow.step
    def run_ph(self):
        """
        Launch the right subworkflow with parallelization or restart
        """
        main_params = self.get_parameters()
        
        use_qgrid = main_params.get('input',{}).get('use_qgrid_parallelization',
                                                    self._use_qgrid_parallelization)
        
        if use_qgrid:
            wf_ph = PhqgridWorkflow(params=main_params)
        else:
            wf_ph = PhrestartWorkflow(params=main_params)
        
        wf_ph.store()
        self.append_to_report("Launching PH sub-workflow (pk: {})".format(wf_ph.pk))
        self.attach_workflow(wf_ph)
        wf_ph.start()
        
        self.next(self.final_step)
        
    @Workflow.step   
    def final_step(self):
        """
        Save results
        """
        main_params = self.get_parameters()
        
        # Retrieve the PH result (calculation or folder)
        wf_ph = self.get_step(self.run_ph).get_sub_workflows()[0]
        
        try:
            ph_calc = wf_ph.get_result('ph_calculation')
            self.add_result("ph_calculation", ph_calc)
        except ValueError:
            # some workflows (parallel cases) do not finish with a PhCalculation
            pass
        ph_folder = wf_ph.get_result('ph_folder')
        self.add_result("ph_folder", ph_folder)

        if 'global_dvscf_folder' in wf_ph.get_results():
            dvscf_folder = wf_ph.get_result('global_dvscf_folder')
            self.add_result("global_dvscf_folder", dvscf_folder)
    
        group_name = main_params.get('group_name',None)
        if group_name is not None:
            # create or get the group, and add the calculation
            group, created = Group.get_or_create(name=group_name)
            if created:
                self.append_to_report("Creating group '{}'".format(group_name))
            self.append_to_report("Adding result to group '{}'"
                                  "".format(group_name))
            group.add_nodes(ph_folder)
        
        self.append_to_report("PH workflow completed")
        
        # clean scratch leftovers, if requested
        if main_params.get('input',{}).get('clean_workdir',self._clean_workdir):
            self.append_to_report("Cleaning scratch directories")
            try:
                save_calcs = [ ph_calc ]
            except NameError:
                save_calcs = []
            helpers.wipe_all_scratch(self, save_calcs)
            
        self.next(self.exit)


class PhqgridWorkflow(Workflow):
    """
    Subworkflow to run QE ph.x code with parallelization on a grid of q-points.
    ph.x is launched on individual q-points, and then the dynamical matrices are 
    collected.
    
    To be called through PhWorkflow.
    """
    try: 
        _default_QE_epsil = PhWorkflow._default_QE_epsil
    except (NameError,AttributeError):
        _default_QE_epsil = False

    def __init__(self,**kwargs):
        super(PhqgridWorkflow, self).__init__(**kwargs)
    
    @Workflow.step
    def start(self):
        """
        Phonon initialization
        """
        self.append_to_report("Starting PH workflow with q-points parallelization")
        
        # runs a fake ph computation (stopped right away) to check the q-points list
        params = self.get_parameters()
        # try first to get such a fake ph calculation that already ran
        old_ph_calcs = get_ph_calc_with_parameters(params, only_initialization=True)
        if old_ph_calcs:
            self.append_to_report("Found {} previous ph.x inititalization"
                                  " calculations".format(len(old_ph_calcs)))
            self.add_attribute('old_phinit_calcs',[_.pk for _ in old_ph_calcs])
        else:
            ph_calc = get_ph_calculation(params, only_initialization=True)
            self.append_to_report("Launching ph.x initialization (pk: {})"
                                  "".format(ph_calc.pk))
            self.attach_calculation(ph_calc)
        self.next(self.run_ph_grid)
        
    @Workflow.step
    def run_ph_grid(self):
        """
        Launch parallel runs on the q-grid
        """
        # runs the parallel ph jobs for each q-point 
        main_params = self.get_parameters()
        
        # Retrieve the list of q-points from the previous step
        ph_init_calc = ([load_node(pk) for pk in self.get_attributes().get(
                            'old_phinit_calcs',[])]
                        + list(self.get_step_calculations(self.start)))[0]
        
        try:
            structure = ph_init_calc.inp.parent_calc_folder.inp.remote_folder.out.output_structure
        except AttributeError:
            structure=ph_init_calc.inp.parent_calc_folder.inp.remote_folder.inp.structure
        
        qpoints_dict = get_distribute_qpoints_results(structure=structure,
                                                 retrieved=ph_init_calc.out.retrieved)

        compute_epsil = main_params['parameters']['INPUTPH'].get('epsil', self._default_QE_epsil)        
        for k in sorted([_ for _ in qpoints_dict if _.count('_')==1]):
            main_params.pop('ph_calculation',None)
            qpoint = qpoints_dict[k]
            
            # Launch the PH computation for this q
            main_params['qpoints'] = qpoint
            
            if not all([_==0. for _ in qpoint.get_kpoints()[0]]):
                # if it's not gamma, we cannot compute the dielectric constant
                main_params['parameters']['INPUTPH']['epsil'] = False
            else:
                # We need to restore the parameter set by the user/default
                main_params['parameters']['INPUTPH']['epsil'] = compute_epsil
            
            # try to find previously run Phrestart workflows with the same parameters
            old_wfs_ph = helpers.get_phrestart_wfs_with_parameters(main_params)
            wfs_ph_completed = sorted([w for w in old_wfs_ph
                                       if 'ph_folder' in w.get_results()],
                                      key=lambda x:x.get_result('ph_folder').ctime)
            if wfs_ph_completed:
                self.append_to_report("Found {} completed previous "
                                      "Phrestart workflows for q-pt {}"
                                      "".format(len(wfs_ph_completed),
                                                qpoint.get_kpoints()))
                self.add_attribute('old_phrestart_completed_wfs_{}'.format(k),
                                   [_.pk for _ in wfs_ph_completed])
            else:
                wfs_ph_cleancalc = sorted([w for w in old_wfs_ph
                                           if 'last_clean_calc' in w.get_attributes()
                                           and (datetime.now(tz=w.get_attribute('last_clean_calc').ctime.tzinfo)-
                                                             w.get_attribute('last_clean_calc').ctime).days<=14],
                                          key=lambda x:x.get_attribute('last_clean_calc').ctime)
                if wfs_ph_cleancalc:
                    self.append_to_report("Found {} unfinished previous "
                                          "Phrestart workflows for q-pt {}"
                                          "".format(len(wfs_ph_cleancalc),
                                                    qpoint.get_kpoints()))
                    self.add_attribute('old_phrestart_cleancalc_wfs_{}'.format(k),
                                       [_.pk for _ in wfs_ph_cleancalc])
                    # we restart from this last clean ph calculation
                    main_params['ph_calculation'] = \
                        wfs_ph_cleancalc[-1].get_attribute('last_clean_calc')
                # launch a new Phrestart workflow
                wf_ph = PhrestartWorkflow(params=main_params)
                wf_ph.store()
                self.append_to_report("Launching Phrestart wf (pk: {})"
                                      " from calc {} for q-pt {}".format(
                                      wf_ph.pk,main_params.get('ph_calculation',
                                            main_params['pw_calculation']).pk,
                                      qpoint.get_kpoints()))
                self.attach_workflow(wf_ph)
                wf_ph.start()

        self.next(self.final_step)
        
    @Workflow.step
    def final_step(self):
        """
        Collect all dynamical matrices back in one folder
        """
        # Find the FolderData with the dynmat-0 file
        ph_init_calc = ([load_node(pk) for pk in self.get_attributes().get(
                            'old_phinit_calcs',[])]
                        + list(self.get_step_calculations(self.start)))[0]
        retrieved_dict = {'initial_folder': ph_init_calc.out.retrieved}
        
        # now the other retrieved folders
        old_ph_workflows = [load_workflow(self.get_attribute(k)[-1])
                            for k in self.get_attributes() 
                            if k.startswith('old_phrestart_completed_wfs_')]
        ph_subworkflows = old_ph_workflows + list(self.get_step(self.run_ph_grid).get_sub_workflows())
        all_retrieved = [_.get_result('ph_folder') for _ in ph_subworkflows]
        all_links = [_.get_parameter('qpoints').get_inputs_dict().keys()[0] 
                     for _ in ph_subworkflows]

        all_remotes = [_.get_result('ph_calculation').out.remote_folder
                       for _ in ph_subworkflows ]

        #rem.out.remote_folder.out.dvscf_remote_folder
        for retrieved, link in zip(all_retrieved, all_links):
            i_point = int(link.split('_')[1])
            retrieved_dict["retrieved_{}".format(i_point)] = retrieved
        
        # Launch the inline calculation to collect the dynamical matrices
        _, ret_dict = recollect_qpoints_inline(**retrieved_dict)
        self.append_to_report("Dynamical matrices recollected")
        
        ph_folder = ret_dict['retrieved']
        self.add_result("ph_folder", ph_folder)
        
        self.append_to_report("PH grid workflow completed (ph folder pk: {})"
                              "".format(ph_folder.pk))


        remotes_dict={}
        for remote, link in zip(all_remotes, all_links):
            i_point = int(link.split('_')[1])
            remotes_dict["remote_{}".format(i_point)] = remote

        params = self.get_parameters()
        dvscf_dir = params.get('input',{}).get('directory_for_dvscf_files',None)
        if dvscf_dir is not None:
            parameters = ParameterData(dict={'destination_directory': dvscf_dir})
            _, global_rem_folder = recollect_allremotes_inline(parameters=parameters,**remotes_dict)
            self.add_result("global_dvscf_folder", global_rem_folder['global_dvscf_folder'])
        
        self.next(self.exit)


class PhrestartWorkflow(Workflow):
    """
    Subworkflow to handle a single QE ph.x run with a restart management in 
    case the wall time is exceeded or other kinds of failures.
    
    To be used through PhWorkflow or PhqgridWorkflow.
    """
    _max_restarts = 5
    _default_alpha_mix_from_QE = 0.7
    
    def __init__(self,**kwargs):
        super(PhrestartWorkflow, self).__init__(**kwargs)
    
    @Workflow.step
    def start(self):
        """
        Just the start
        """
        self.append_to_report("Starting PH restart workflow")
        self.next(self.run_ph_restart)
        
    @Workflow.step
    def run_ph_restart(self):
        """
        Looped step
        """
        
        # launch Phonon code, or restart it if maximum wall time was exceeded,
        # or if ph did not reach the normal end of execution.
        # go to final_step when computation succeeded in previous step.
        
        # retrieve PH parameters
        params = self.get_parameters()
        
        # Retrieve the list of PH calculations already done in this step
        ph_calc_list = list(self.get_step_calculations(self.run_ph_restart).order_by('ctime'))
            
        # retrieve attributes
        attr_dict = self.get_attributes()
        # check if previous calculation has failed unexpectedly (not due to time
        # limit nor with the warning 'QE ph run did not reach the end of the 
        # execution.')
        # when has_failed is True, we try to relaunch again ONLY ONCE
        has_failed = attr_dict.get('has_failed',False)
        submission_has_failed = attr_dict.get('submission_has_failed',False)
        # parameters to update, in case they need to be changed after this step
        update_params = helpers.default_nested_dict()
        
        has_finished = False
 
        if ph_calc_list: # if there is at least one previous calculation
            # Analyses what happened with the previous calculation
            
            # Retrieve the last PH calculation done inside this subworkflow
            ph_calc = ph_calc_list[-1]
            
            # test if it needs to be restarted
            if ph_calc.has_finished_ok():
                # computation succeeded -> go to final step
                # ph_calc is the new last clean calculation
                self.add_attribute('last_clean_calc',ph_calc)  
                # reinitialize has_failed and submission_has_failed attributes
                self.add_attribute('has_failed',False)
                self.add_attribute('submission_has_failed',False)
                
                has_finished = True
                                
            if ph_calc.get_state() in [calc_states.SUBMISSIONFAILED]:
                # case when submission failed, probably due to an IO error
                # -> try to restart once
                if submission_has_failed:
                    # this is already the second time the same submission
                    # fails -> workflow stops
                    self.append_to_report("ERROR: ph.x (pk: {0}) submission failed "
                                          "unexpectedly a second time".format(ph_calc.pk))
                    raise ValueError("ERROR: submission failed twice")
                else:
                    # first time this happens -> try to re-submit once
                    self.append_to_report("WARNING: ph.x (pk: {}) submission failed "
                                          "unexpectedly, will try to restart"
                                          " once again".format(ph_calc.pk))
                    # set submission_has_failed attribute
                    self.add_attribute('submission_has_failed',True)
                    
            else:
                # submission did not fail -> reinitialize the
                # submission_has_failed attribute
                self.add_attribute('submission_has_failed',False)
                
            if ph_calc.get_state() in [calc_states.FAILED]:
                if 'Maximum CPU time exceeded' in ph_calc.res.warnings:
                    # maximum CPU time was exceeded -> will restart
                    self.append_to_report("ph.x calculation (pk: {0}) stopped "
                                          "because max CPU time was "
                                          "reached; restarting computation "
                                          "where it stopped"
                                          "".format(ph_calc.pk))
                    # ph_calc is the new last clean calculation
                    self.add_attribute('last_clean_calc',ph_calc)
                    # reinitialize has_failed attribute (here it is not a 
                    # "real" failure)
                    self.add_attribute('has_failed',False)

                elif ( ("QE ph run did not reach the end of the execution." in 
                       ph_calc.res.parser_warnings) and len(ph_calc.res.warnings)==0 ):
                    # case of an unexpected stop during an scf step -> try again
                    # from the last clean calculation
                    max_sec = ph_calc.inp.parameters.get_dict()['INPUTPH']['max_seconds']
                    self.append_to_report("WARNING: ph.x (pk: {}) did not reach "
                                          "end of execution -> will try to restart"
                                          " with max_seconds={}".format(ph_calc.pk,
                                                                       int(max_sec*0.95)))
                    # Note: ph_calc is not a clean calculation, so we do
                    # not update the 'last_clean_calc' attribute

                    # we reduce the max wall time in the pw input file, to avoid
                    # stopping in the middle of an scf step
                    update_params['parameters']['INPUTPH']['max_seconds'] = int(max_sec*0.95)

                elif any(["read_namelists" in w 
                                        for w in ph_calc.res.warnings]):   
                    # any other case leads to stop on error message
                    self.append_to_report("ERROR: incorrect input file for ph.x "
                                          "calculation (pk: {0}) , stopping; "
                                          "list of warnings:\n {1}".format(
                                                ph_calc.pk,ph_calc.res.warnings))
                    raise ValueError("ERROR: incorrect input parameters")
            
                elif ( "Phonon did not reach end of self consistency" in 
                       ph_calc.res.warnings ):
                    # case of a too slow scf convergence -> try again with 
                    # smaller mixing beta
                    alpha_mix = ph_calc.inp.parameters.get_dict()['INPUTPH'].get(
                                    'alpha_mix(1)',self._default_alpha_mix_from_QE)
                    self.append_to_report("WARNING: ph.x (pk: {}) scf convergence"
                                          " too slow -> will try with "
                                          "alpha_mix={}".format(ph_calc.pk,
                                                                alpha_mix*0.9))
                    # ph_calc is the new last clean calculation
                    self.add_attribute('last_clean_calc',ph_calc)
                    
                    # update parameters
                    update_params["parameters"]["INPUTPH"]["alpha_mix(1)"] = alpha_mix*0.9
                    
                    # reinitialize has_failed attribute (here it is not a 
                    # "real" failure)
                    self.add_attribute('has_failed',False)

                else:
                    # case of a real failure
                    if has_failed:
                        # this is already the second time the same calculation
                        # fails -> workflow stops
                        self.append_to_report("ERROR: ph.x (pk: {0}) failed "
                                              "unexpectedly a second time-> "
                                              "stopping;\n list of warnings:\n {1} "
                                              "\n list of parser warnings:\n {2}"
                                              "".format(ph_calc.pk,
                                                        ph_calc.res.warnings,
                                                        ph_calc.res.parser_warnings))
                        raise Exception("ERROR: the same calculation failed "
                                        "twice")
                    else:
                        # first time this happens -> try to restart from the 
                        # last clean calculation
                        self.append_to_report("WARNING: ph.x (pk: {}) failed "
                                              "unexpectedly, will try to restart"
                                              " once again".format(ph_calc.pk))
                        # set has_failed attribute
                        self.add_attribute('has_failed',True)
                        # Note: ph_calc is not a clean calculation, so we do
                        # not update the 'last_clean_calc' attribute
                
            if (ph_calc.get_state() not in [calc_states.FINISHED, 
                                            calc_states.FAILED, 
                                            calc_states.SUBMISSIONFAILED]):   
                # any other case leads to stop on error message
                self.append_to_report("ERROR: unexpected state ({0}) of ph.x "
                                      "(pk: {1}) calculation, stopping"
                                      "".format(ph_calc.get_state(),ph_calc.pk))
                raise Exception("ERROR: unexpected state")

        
        # decide what to do next
        if has_finished:
            self.next(self.final_step)
        
        else:        
            # Stop if we reached the max. number of restarts
            if len(ph_calc_list) >= params.get('input',{}).get('max_restarts',
                                                     self._max_restarts):
                # when we exceed the max. number of restarts -> stop on error message
                self.append_to_report("ERROR: Max number of restarts reached "
                                      "(last calc={})".format(ph_calc.pk))
                raise Exception("ERROR: maximum number of restarts reached "
                                "(increase 'max_restarts')")
        
            # retrieve attributes again
            attr_dict = self.get_attributes()
            old_update_params = attr_dict.get('update_params',{})
            # new set of parameters to update
            update_params = helpers.update_nested_dict(old_update_params,update_params)
            self.add_attribute('update_params',update_params)
            # new parameters
            params = helpers.update_nested_dict(params,update_params)
            
            if attr_dict.get('last_clean_calc',None) is None:
                # Initial PH computation
                
                # Launch the phonon code (first trial, or start from a 
                # former ph calc from the parameters)
                parent_calc = params.get('ph_calculation',None)
                ph_calc = get_ph_calculation(params,parent_calc=parent_calc)
                self.append_to_report("Launching ph.x (pk: {}){}".format(ph_calc.pk,
                        " from former ph calculation {}".format(parent_calc.pk)
                        if parent_calc else ""))
                self.attach_calculation(ph_calc)
                
                # loop step on itself
                self.next(self.run_ph_restart)
            
            else:
                # Restart PH computation
                # retrieve last clean ph calculation
                last_clean_ph_calc = self.get_attribute('last_clean_calc')
                # prepare restarted calculation
                ph_new_calc = get_ph_calculation(params,parent_calc=last_clean_ph_calc)
                # Launch restarted calculation
                self.append_to_report("Launching ph.x (pk: {}) from previous "
                                      "calculation (pk: {})".format(ph_new_calc.pk,
                                                                   last_clean_ph_calc.pk))
                self.attach_calculation(ph_new_calc)

                # loop step on itself
                self.next(self.run_ph_restart)
                
    @Workflow.step   
    def final_step(self):
        """
        Create the results
        """
        params = self.get_parameters()
        ph_calc_list = list(self.get_step_calculations(self.run_ph_restart).order_by('ctime'))
        ph_calc = ph_calc_list[-1]
        self.add_result("ph_calculation", ph_calc)
        self.add_result("ph_folder", ph_calc.out.retrieved)

        dvscf_dir = params.get('input',{}).get('directory_for_dvscf_files',None)
        if dvscf_dir is not None:
            parameters = ParameterData(dict={'destination_directory': dvscf_dir})
            _, result_dict = copy_dvscf_files_inline(parameters=parameters,
                                                     remote_folder=ph_calc.out.remote_folder)
            self.append_to_report("Density files copied into {} (RemoteData pk: "
                                  "{})".format(dvscf_dir,
                                              result_dict['dvscf_remote_folder'].pk))


        
        self.append_to_report("PH restart workflow completed")
        self.next(self.exit)

