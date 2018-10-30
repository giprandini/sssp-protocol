# -*- coding: utf-8 -*-
from aiida.orm.workflow import Workflow
from aiida.orm import CalculationFactory, Code, DataFactory, Group, load_workflow
from aiida.workflows.user.epfl_theos.quantumespresso.pw import PwWorkflow
from aiida.workflows.user.epfl_theos.quantumespresso import helpers
from aiida.workflows.user.epfl_theos.quantumespresso.ph import PhWorkflow
from aiida.common.example_helpers import test_and_get_code
from aiida.orm.data.array.kpoints import _default_epsilon_length,_default_epsilon_angle
from datetime import datetime

__copyright__ = u"Copyright (c), This file is part of the AiiDA-EPFL Pro platform. For further information please visit http://www.aiida.net/. All rights reserved"
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file."
__version__ = "0.1.0"
__authors__ = "Nicolas Mounet, Andrea Cepellotti, Giovanni Pizzi."


ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
StructureData = DataFactory('structure')
PwCalculation = CalculationFactory('quantumespresso.pw')
PhCalculation = CalculationFactory('quantumespresso.ph')

class PhonondispersionWorkflow(Workflow):
    """
    Workflow to compute the phonon dispersion from the raw initial unrelaxed
    structure.
    
    Results posted in results depend on the input parameters. 
    The largest set of data that can be put in the results consists in:
    * relaxed structure, if relaxed
    * pw_calculation, if the total energy needed to be computed
    * band_structure or band_structure1, band_structure2 (for spin polarized 
      calculations), containing the electronic band structure
    * ph_folder, i.e. the folder with all dynamical matrices
    * ph_calculation, if the phonon calculation was not parallelized over qpoints
    * phonon_dispersion, a BandsData object
    
    Input description.
    The input follows closely the input of the various subworkflows
    
    Electronic part:
    'structure': structure,
    'pseudo_family': pseudo_family,    
    'pw_codename': pw_codename,
    'pw_settings': settings,
    'pw_parameters': pw_input_dict,
    'pw_calculation_set': set_dict,
    'pw_kpoints': kpoints,
    'pw_input':{'relaxation_scheme': relaxation_scheme,
                  'volume_convergence_threshold': 1.e-2,
                  },

    OR

    'pw_calculation': load_node(60),


     'ph_calculation': load_node(157),
     
     'ph_codename': ph_codename,
     'ph_parameters': ph_input_dict,
     'ph_settings': settings,
     'ph_calculation_set': set_dict,
     'ph_qpoints': qpoints,
     'ph_input': {'use_qgrid_parallelization': True},
     
     'dispersion_matdyn_codename': matdyn_codename,
     'dispersion_q2r_codename': q2r_codename,
     'dispersion_calculation_set': set_dict,
     'dispersion_settings': settings,
     'dispersion_input':{'distance_qpoints_in_dispersion':0.01,
                   'asr': asr,
                   'zasr': asr,
                   'threshold_length_for_Bravais_lat': 1e-4,
                   'threshold_angle_for_Bravais_lat': 1e-4,
                   }
    """
    
    _qpoint_distance_in_dispersion = 0.01
    _default_epsilon_length = _default_epsilon_length
    _default_epsilon_angle = _default_epsilon_angle
    _clean_workdir = False
    
    def __init__(self,**kwargs):
        super(PhonondispersionWorkflow, self).__init__(**kwargs)
    
    @Workflow.step
    def start(self):
        """
        Checks the parameters
        """
        self.append_to_report("Checking input parameters")
        
        mandatory_pw_keys = [('structure',StructureData,"the structure (a previously stored StructureData object)"),
                             ('pseudo_family',basestring,'the pseudopotential family'),
                             ('pw_kpoints',KpointsData,'A KpointsData object with the kpoint mesh used by PWscf'),
                             #('pw_calculation_set',dict,'A dictionary with resources, walltime, ...'),
                             ('pw_parameters',dict,"A dictionary with the PW input parameters"),
                             ]
        mandatory_ph_keys = [('ph_qpoints',KpointsData,'A KpointsData object with the kpoint mesh used by PWscf'),
                             ('ph_calculation_set',dict,'A dictionary with resources, walltime, ...'),
                             ('ph_parameters',dict,"A dictionary with the PH input parameters"),
                             ]
        mandatory_dispersion_keys=[('dispersion_calculation_set', dict,'A dictionary with resources, walltime, ...'),
                             ]
                
        main_params = self.get_parameters()
        
        # validate the codes
        for kind,key in [['quantumespresso.pw','pw_codename'],
                         ['quantumespresso.ph','ph_codename'],
                         ['quantumespresso.q2r','dispersion_q2r_codename'],
                         ['quantumespresso.matdyn','dispersion_matdyn_codename'],
                         ]:
            try:
                test_and_get_code(main_params[key], kind, use_exceptions=True)
            except KeyError:
                # none of the codes is always required
                pass
        
        # case of restart from phonon calculation
        if 'ph_calculation' in main_params:
            if isinstance(main_params['ph_calculation'], PhCalculation):
                helpers.validate_keys(main_params, mandatory_dispersion_keys)
                self.next(self.run_q2r)
                return
            else:
                raise TypeError("parameter 'ph_calculation' should be a "
                                "PhCalculation")
        
        # case of restart from phonon folder
        if 'ph_folder' in main_params:
            if isinstance(main_params['ph_folder'], DataFactory('folder')):
                helpers.validate_keys(main_params, mandatory_dispersion_keys)
                self.next(self.run_q2r)
                return
            else:
                raise TypeError("parameter 'ph_folder' should be a FolderData")
        
        # validate phonon keys
        helpers.validate_keys(main_params, mandatory_ph_keys)
        
        if 'pw_calculation' in main_params:
            if isinstance(main_params['pw_calculation'],PwCalculation):
                # if pw is a calculation, launch directly from the PH step
                self.next(self.run_ph)
                return
            else:
                raise TypeError("parameter 'pw_calculation' should be a "
                                "PwCalculation")
    
        # validate Pw keys
        helpers.validate_keys(main_params, mandatory_pw_keys)
        
        # start from Pw calculation
        self.next(self.run_pw)
    
                
    @Workflow.step
    def run_pw(self):
        """
        Launch the PwWorkflow
        """
        main_params = self.get_parameters()
        # take the parameters needed for the PW computation
        pw_params = {}
        for k,v in main_params.iteritems():
            if k.startswith('ph_') or k.startswith('dispersion_'):
                pass
            elif k.startswith('pw_'):
                new_k = k[3:] # remove pw_
                pw_params[new_k] = v
            else:
                pw_params[k] = v
        
        # I enforce always a final SCF (if not specified), and if needed I remove 'use_all_frac'
        # (the latter used only if finish_with_scf = True
        pw_params['input'] = pw_params.get('input', {})
        pw_params['input']['finish_with_scf'] = pw_params['input'].get('finish_with_scf', True)
        pw_params = helpers.update_nested_dict(pw_params,{'finalscf_parameters_update':{'SYSTEM':{'use_all_frac': False}}})
        #pw_params['input']['final_scf_remove_use_all_frac'] = True # DEPRECATED
        
        # try to find previously run Pw wfs with the same parameters
        old_wfs_pw = [w for w in helpers.get_pw_wfs_with_parameters(pw_params)
                      if (datetime.now(tz=w.get_result('pw_calculation').ctime.tzinfo)-
                                       w.get_result('pw_calculation').ctime).days<=14]
        if old_wfs_pw:
            self.append_to_report("Found {} completed previous "
                                  "Pw workflows"
                                  "".format(len(old_wfs_pw)))
            self.add_attribute('old_pw_wfs',[_.pk for _ in old_wfs_pw])
        else:
            wf_pw = PwWorkflow(params=pw_params)        
            wf_pw.store()
            self.append_to_report("Launching PW sub-workflow (pk: {})".format(wf_pw.pk))
            self.attach_workflow(wf_pw)
            wf_pw.start()
        
        self.next(self.run_ph)
        
        
    @Workflow.step
    def run_ph(self):
        main_params = self.get_parameters()
        
        try:
            pw_calculation = main_params['pw_calculation']
        except KeyError:
            wf_pw = ([load_workflow(pk) for pk in self.get_attributes(
                ).get('old_pw_wfs',[])] + list(self.get_step(self.run_pw).get_sub_workflows()))[0]
            pw_calculation = wf_pw.get_result('pw_calculation')
            
            # Save results of the subworkflow in the main
            for k,v in wf_pw.get_results().iteritems():
                self.add_result(k,v)
        
        # load the PhWorkflow parameters
        
        ph_params = {'pw_calculation': pw_calculation}
        for k,v in main_params.iteritems():
            if k.startswith('ph_'):
                new_k = k[3:] # remove ph_
                ph_params[new_k] = v
        
        # deactivate dielectric constant calculation if pw calculation done
        # with a smearing (metal)
        if pw_calculation.res.smearing_method:
            ph_params['parameters']['INPUTPH']['epsil'] = False
            self.append_to_report("Pw calculation (pk: {}) done with smearing -> "
                                  "forcing epsil=False for Ph calculation".format(pw_calculation.pk))
        
        # Launch the PH sub-workflow
        wf_ph = PhWorkflow(params=ph_params)
        wf_ph.store()
        self.append_to_report("Launching PH sub-workflow (pk: {})".format(wf_ph.pk))
        self.attach_workflow(wf_ph)
        wf_ph.start()
        
        if any( [ k.startswith('dispersion') for k in main_params.keys() ] ):
            self.next(self.run_q2r)
            return
        else:
            self.next(self.final_step_ph)
            return
    
    @Workflow.step
    def final_step_ph(self):
        """
        This is the final step if you do not call q2r and matdyn
        """
        main_params = self.get_parameters()
        
        # Retrieve the MATDYN calculation  
        ph_wf = self.get_step(self.run_ph).get_sub_workflows()[0]
        ph_wf_res = ph_wf.get_results()
        
        for k,v in ph_wf_res.iteritems():
            self.add_result(k,v)
        
        self.append_to_report("Phonons computed and put in results")
        
        # clean scratch leftovers, if requested
        if main_params.get('input',{}).get('clean_workdir',self._clean_workdir):
            self.append_to_report("Cleaning scratch directories")
            save_calcs = []
            try:
                # Note that the order is important!
                save_calcs.append( self.get_result('ph_calculation') )
                save_calcs.append( self.get_result('pw_calculation') )
            except (NameError, ValueError):
                pass
            helpers.wipe_all_scratch(self, save_calcs)

        self.next(self.exit)
    
    @Workflow.step
    def run_q2r(self):
        """
        Launch q2r
        """
        main_params = self.get_parameters()
        
        # load the parent phonon
        if 'ph_calculation' in main_params:
            # PhCalculation passed in input
            ph_calculation = main_params['ph_calculation']
            ph_folder = ph_calculation.out.retrieved
        elif 'ph_folder' in main_params:
            # PhCalculation.retrieved passed in input
            ph_folder = main_params['ph_folder']
        else:
            # The PhCalculation was run in the previous steps
            wf_ph = self.get_step(self.run_ph).get_sub_workflows()[0]
            ph_folder = wf_ph.get_result('ph_folder')
            
            # Save results of the subworkflow in the main
            for k,v in wf_ph.get_results().iteritems():
                self.add_result(k,v)
        
        # Launch the Q2R computation
        try: 
            q2r_parameters = ParameterData(dict={'INPUT': 
                                                 {'zasr': main_params['dispersion_input']["zasr"],
                                                  'do_cutoff_2D': main_params['dispersion_input']["do_cutoff_2D"]}})
        except KeyError:
            q2r_parameters = ParameterData(dict={'INPUT': 
                                                 {'zasr': main_params['dispersion_input']["zasr"]}})
        code = Code.get_from_string(main_params["dispersion_q2r_codename"])
        q2r_calc = code.new_calc()
        q2r_calc.use_parameters(q2r_parameters)
        q2r_calc.use_parent_folder(ph_folder)
        q2r_calc = helpers.set_the_set(q2r_calc, 
                                       main_params['dispersion_calculation_set'])
        q2r_calc.store_all()
        
        self.append_to_report("Launching Q2R (pk: {})".format(q2r_calc.pk))
        self.attach_calculation(q2r_calc)
        
        self.next(self.run_matdyn)


    @Workflow.step
    def run_matdyn(self):
        """
        Launch matdyn
        """
        main_params = self.get_parameters()
        
        # Retrieve the Q2R calculation
        q2r_calc = self.get_step_calculations(self.run_q2r)[0]
        
        # Launch the MATDYN computation
        dispersion_input = main_params.get("dispersion_input",{})
        
        # extract cell and periodic boundary conditions from force constants
        force_constants = q2r_calc.out.force_constants
        cell = force_constants.cell
        pbc = tuple( _ > 1 for _ in force_constants.qpoints_mesh)

        # set the qpoints
        try:
            qpoints = main_params["dispersion_qpoints"]
        except KeyError:
            qpoints = KpointsData()
            qpoint_distance = dispersion_input.get("distance_qpoints_in_dispersion",
                                               self._qpoint_distance_in_dispersion)
            epsilon_length = dispersion_input.get("threshold_length_for_Bravais_lat",
                                               self._default_epsilon_length)
            epsilon_angle = dispersion_input.get("threshold_angle_for_Bravais_lat",
                                               self._default_epsilon_angle)
            qpoints.set_cell(cell,pbc)
            qpoints.set_kpoints_path(value=dispersion_input.get("qpoints_path",None),
                                     kpoint_distance=qpoint_distance,
                                     epsilon_length=epsilon_length,
                                     epsilon_angle=epsilon_angle)
        try: 
            matdyn_parameters = ParameterData(dict={'INPUT': 
                                                    {'asr': dispersion_input['asr'],
                                                     'do_cutoff_2D': dispersion_input['do_cutoff_2D']}})
        except KeyError:
            matdyn_parameters = ParameterData(dict={'INPUT': {'asr': 
                                                              dispersion_input['asr']}})
        
        code = Code.get_from_string(main_params["dispersion_matdyn_codename"])
        matdyn_calc = code.new_calc()
        matdyn_calc.use_parameters(matdyn_parameters)
        matdyn_calc.use_kpoints(qpoints)
        matdyn_calc.use_parent_calculation(q2r_calc)
        matdyn_calc = helpers.set_the_set(matdyn_calc, 
                                          main_params['dispersion_calculation_set'])
        if 'dispersion_settings' in main_params:
            matdyn_calc.use_settings(ParameterData(dict=main_params['dispersion_settings']))
        matdyn_calc.store_all()
        
        self.append_to_report("Launching MATDYN (pk: {})".format(matdyn_calc.pk))
        self.attach_calculation(matdyn_calc)
        
        self.next(self.final_step)
    
    
    @Workflow.step
    def final_step(self):
        """
        Append results
        """
        main_params = self.get_parameters()
        
        # Retrieve the MATDYN calculation  
        matdyn_calc = self.get_step_calculations(self.run_matdyn)[0]
        
        # get dispersions
        bandsdata = matdyn_calc.out.output_phonon_bands
        
        self.append_to_report("Phonon dispersions done (bandsdata pk: {})"
                              "".format(bandsdata.pk))
        
        self.add_result("phonon_dispersion", bandsdata)
        bandsdata.label = "Phonon bands"
        bandsdata.description = ("Phonon dispersion calculated with"
                                 " the workflow {}".format(self.pk))
        
        group_name = main_params.get('dispersion_group_name',None)
        if group_name is not None:
            # create or get the group
            group, created = Group.get_or_create(name=group_name)
            if created:
                self.append_to_report("Created group '{}'".format(group_name))
            # put the bands data into the group
            group.add_nodes(bandsdata)
            self.append_to_report("Adding bands to group '{}'".format(group_name))
        
        # clean scratch leftovers, if requested
        if main_params.get('dispersion_input',{}).get('clean_workdir',self._clean_workdir):
            self.append_to_report("Cleaning scratch directories")
            save_calcs = []
            try:
                # Note that the order is important!
                save_calcs.append( self.get_result('ph_calculation') )
                save_calcs.append( self.get_result('pw_calculation') )
            except (NameError, ValueError):
                pass
            helpers.wipe_all_scratch(self, save_calcs)
            
        self.next(self.exit)
