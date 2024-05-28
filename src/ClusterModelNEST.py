import numpy as np
from copy import deepcopy
import time
import pickle
import sys
import signal
sys.path.append("..")
import ClusterHelper
import GeneralHelper
import nest


class ClusteredNetworkBase:
    """ 
    Creates an object with functions to create neuron populations, 
    stimulation devices and recording devices for an EI-clustered network.
    Provides also function to initialize NEST (v3.x), simulate the network and
    to grab the spike data.
    """

    def __init__(self, defaultValues, parameters):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and 
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network.

        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        parameters = deepcopy(parameters)
        self.params = GeneralHelper.mergeParams(parameters, defaultValues)
        self.Populations = []
        self.RecordingDevices = []
        self.Currentsources = []

    def clean_network(self):
        """
        Creates empty attributes of a network.
        """
        self.Populations = []
        self.RecordingDevices = []
        self.Currentsources = []

    def setup_nest(self):
        """ Initializes the NEST kernel.
        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual used one if none is supplied.
        """
        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')
        self.params['randseed'] = self.params.get('randseed',
                                        np.random.randint(1000000))


        np.random.seed(self.params['randseed'])
        randseeds = list(range(self.params['randseed'] + 2, 
                               self.params['randseed'] + 2 + self.params.get('n_jobs', 1)))
        nest.SetKernelStatus({"resolution": self.params.get('dt', 0.1),
                              "print_time": True, "overwrite_files": True,
                              'local_num_threads': self.params.get('n_jobs', 1),
                              'grng_seed': self.params['randseed']  + 1,
                              'rng_seeds': randseeds})

    def create_populations(self):
        """
        Creates Q excitatory and inhibitory neuron populations with the parameters of the network.

        """
        # make sure number of clusters and units are compatible
        #assert self.params['N_E'] % self.params['Q'] == 0, 'N_E needs to be evenly divisible by Q'
        #assert self.params['N_I'] % self.params['Q'] == 0, 'N_I needs to be evenly divisible by Q'

        N = self.params['N_E'] + self.params['N_I']  # total units
        try:
            DistParams = self.params['DistParams']
        except AttributeError:
            DistParams = {'distribution': 'normal', 'sigma': 0.0, 'fraction': False}

        if self.params['I_th_E'] is None:
            I_xE = self.params['I_xE']
        else:
            I_xE = self.params['I_th_E'] * (self.params['V_th_E'] - self.params['E_L']) / self.params['tau_E'] * \
                   self.params['C_m']

        if self.params['I_th_I'] is None:
            I_xI = self.params['I_xI']
        else:
            I_xI = self.params['I_th_I'] * (self.params['V_th_I'] - self.params['E_L']) / self.params['tau_I'] * \
                   self.params['C_m']

        E_neuron_params = {'E_L': self.params['E_L'], 'C_m': self.params['C_m'], 'tau_m': self.params['tau_E'],
                           't_ref': self.params['t_ref'], 'V_th': self.params['V_th_E'], 'V_reset': self.params['V_r'],
                           'I_e': I_xE}
        I_neuron_params = {'E_L': self.params['E_L'], 'C_m': self.params['C_m'], 'tau_m': self.params['tau_I'],
                           't_ref': self.params['t_ref'], 'V_th': self.params['V_th_I'], 'V_reset': self.params['V_r'],
                           'I_e': I_xI}
        if 'iaf_psc_exp' in self.params['neuron_type']:
            E_neuron_params['tau_syn_ex'] = self.params['tau_syn_ex']
            E_neuron_params['tau_syn_in'] = self.params['tau_syn_in']
            I_neuron_params['tau_syn_in'] = self.params['tau_syn_in']
            I_neuron_params['tau_syn_ex'] = self.params['tau_syn_ex']

            # iaf_psc_exp allows stochasticity, if not used - ignore
            try:
                if self.params['delta_'] is not None:
                    E_neuron_params['delta'] = self.params['delta_']
                    I_neuron_params['delta'] = self.params['delta_']
                if self.params['rho'] is not None:
                    E_neuron_params['rho'] = self.params['rho']
                    I_neuron_params['rho'] = self.params['rho']
            except KeyError:
                pass
        else:
            assert 'iaf_psc_exp' in self.params['neuron_type'], "iaf_psc_exp neuron model is the only implemented model"

            # create the neuron populations
        E_pops = []
        I_pops = []
        for q in range(self.params['Q']):
            E_pops.append(nest.Create(
                self.params['neuron_type'], 
                int(self.params['N_E'] / self.params['Q'])))
            nest.SetStatus(E_pops[-1], E_neuron_params)
            
        # check if the model is E or EI clustered
        jipfactor = self.params.get('jipfactor',None)
        if jipfactor==0.:
            I_pops = [nest.Create(self.params['neuron_type'], 
                                int(self.params['N_I']))]
            nest.SetStatus(I_pops[-1], I_neuron_params)         
        else:
            for q in range(self.params['Q']):
                I_pops.append(
                    nest.Create(self.params['neuron_type'], 
                                int(self.params['N_I'] / self.params['Q'])))
                nest.SetStatus(I_pops[-1], I_neuron_params)

        if self.params['delta_I_xE'] > 0:
            for E_pop in E_pops:
                I_xEs = nest.GetStatus(E_pop, 'I_e')
                nest.SetStatus(E_pop, [
                    {'I_e': (1 - 0.5 * self.params['delta_I_xE'] + np.random.rand() * self.params['delta_I_xE']) * ixe}
                    for ixe in I_xEs])

        if self.params['delta_I_xI'] > 0:
            for I_pop in I_pops:
                I_xIs = nest.GetStatus(I_pop, 'I_e')
                nest.SetStatus(I_pop, [
                    {'I_e': (1 - 0.5 * self.params['delta_I_xI'] + np.random.rand() * self.params['delta_I_xI']) * ixi}
                    for ixi in I_xIs])
        if self.params['V_m'] == 'rand':
            T_0_E = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                             self.params['C_m'], self.params['V_th_E'],
                                                             self.params['V_r'])
            if np.isnan(T_0_E):
                T_0_E = 10.
            for E_pop in E_pops:
                nest.SetStatus(E_pop, [{'V_m': ClusterHelper.V_FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                                   self.params['C_m'], T_0_E * np.random.rand(),
                                                                   self.params['V_th_E'], self.params['t_ref'])} for i
                                       in range(len(E_pop))])

            T_0_I = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                             self.params['C_m'], self.params['V_th_I'],
                                                             self.params['V_r'])
            if np.isnan(T_0_I):
                T_0_I = 10.
            for I_pop in I_pops:
                nest.SetStatus(I_pop, [{'V_m': ClusterHelper.V_FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                                   self.params['C_m'], T_0_I * np.random.rand(),
                                                                   self.params['V_th_E'], self.params['t_ref'])} for i
                                       in range(len(I_pop))])
        else:
            nest.SetStatus(nest.NodeCollection([x for x in range(1, N + 1)]),
                           [{'V_m': self.params['V_m']} for i in range(N)])
        self.Populations = [E_pops, I_pops]

    def connect(self):
        """ Connects the excitatory and inhibitory populations with each other in the EI-clustered scheme
        """
        #  self.Populations[0] -> Excitatory population
        #  self.Populations[1] -> Inhibitory population
        # connectivity parameters
        js = self.params['js']  # connection weights
        N = self.params['N_E'] + self.params['N_I']  # total units

        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        if np.isnan(js).any():
            js = ClusterHelper.calc_js(self.params)
            self.params['js'] = js
        js *= self.params['s']

        # jminus is calculated so that row sums remain constant
        if self.params['Q'] > 1:
            jminus = (self.params['Q'] - self.params['jplus']) / float(self.params['Q'] - 1)
        else:
            self.params['jplus'] = np.ones((2, 2))
            jminus = np.ones((2, 2))

        # define the synapses and connect the populations
        # EE
        j_ee = js[0, 0] / np.sqrt(N)
        nest.CopyModel("static_synapse", "EE_plus",
                       {"weight": self.params['jplus'][0, 0] * j_ee, "delay": self.params['delay']})
        nest.CopyModel("static_synapse", "EE_minus", {"weight": jminus[0, 0] * j_ee, "delay": self.params['delay']})
        if self.params['fixed_indegree']:
            K_EE = int(self.params['ps'][0, 0] * self.params['N_E'] / self.params['Q'])
            print('K_EE: ', K_EE)
            conn_params_EE = {'rule': 'fixed_indegree', 'indegree': K_EE, 'autapses': False,
                              'multapses': False}

        else:
            conn_params_EE = {'rule': 'pairwise_bernoulli', 'p': self.params['ps'][0, 0], 'autapses': False,
                              'multapses': False}
        for i, pre in enumerate(self.Populations[0]):
            for j, post in enumerate(self.Populations[0]):
                if i == j:
                    # same cluster
                    nest.Connect(pre, post, conn_params_EE, 'EE_plus')
                else:
                    nest.Connect(pre, post, conn_params_EE, 'EE_minus')

        # EI
        j_ei = js[0, 1] / np.sqrt(N)
        nest.CopyModel("static_synapse", "EI_plus",
                       {"weight": j_ei * self.params['jplus'][0, 1], 
                        "delay": self.params['delay']})
        nest.CopyModel("static_synapse", "EI_minus", 
                       {"weight": j_ei * jminus[0, 1], 
                        "delay": self.params['delay']})
        if self.params['fixed_indegree']:
            K_EI = int(self.params['ps'][0, 1] * self.params['N_I'] / self.params['Q'])
            print('K_EI: ', K_EI)
            conn_params_EI = {'rule': 'fixed_indegree', 
                              'indegree': K_EI, 'autapses': False,
                              'multapses': False}
        else:
            conn_params_EI = {'rule': 'pairwise_bernoulli', 
                              'p': self.params['ps'][0, 1], 'autapses': False,
                              'multapses': False}
        for i, pre in enumerate(self.Populations[1]):
            for j, post in enumerate(self.Populations[0]):
                if i == j:
                    # same cluster
                    nest.Connect(pre, post, conn_params_EI, 'EI_plus')
                else:
                    nest.Connect(pre, post, conn_params_EI, 'EI_minus')
        # IE
        j_ie = js[1, 0] / np.sqrt(N)
        nest.CopyModel("static_synapse", "IE_plus",
                       {"weight": j_ie * self.params['jplus'][1, 0], "delay": self.params['delay']})
        nest.CopyModel("static_synapse", "IE_minus", {"weight": j_ie * jminus[1, 0], "delay": self.params['delay']})

        if self.params['fixed_indegree']:
            K_IE = int(self.params['ps'][1, 0] * self.params['N_E'] / self.params['Q'])
            print('K_IE: ', K_IE)
            conn_params_IE = {'rule': 'fixed_indegree', 'indegree': K_IE, 'autapses': False,
                              'multapses': False}
        else:
            conn_params_IE = {'rule': 'pairwise_bernoulli', 'p': self.params['ps'][1, 0], 'autapses': False,
                              'multapses': False}
        for i, pre in enumerate(self.Populations[0]):
            for j, post in enumerate(self.Populations[1]):
                if i == j:
                    # same cluster
                    nest.Connect(pre, post, conn_params_IE, 'IE_plus')
                else:
                    nest.Connect(pre, post, conn_params_IE, 'IE_minus')

        # II
        j_ii = js[1, 1] / np.sqrt(N)
        nest.CopyModel("static_synapse", "II_plus",
                       {"weight": j_ii * self.params['jplus'][1, 1], "delay": self.params['delay']})
        nest.CopyModel("static_synapse", "II_minus", {"weight": j_ii * jminus[1, 1], "delay": self.params['delay']})
        if self.params['fixed_indegree']:
            K_II = int(self.params['ps'][1, 1] * self.params['N_I'] / self.params['Q'])
            print('K_II: ', K_II)
            conn_params_II = {'rule': 'fixed_indegree', 'indegree': K_II, 'autapses': False,
                              'multapses': False}
        else:
            conn_params_II = {'rule': 'pairwise_bernoulli', 'p': self.params['ps'][1, 1], 'autapses': False,
                              'multapses': False}
        for i, pre in enumerate(self.Populations[1]):
            for j, post in enumerate(self.Populations[1]):
                if i == j:
                    # same cluster
                    nest.Connect(pre, post, conn_params_II, 'II_plus')
                else:
                    nest.Connect(pre, post, conn_params_II, 'II_minus')


    def create_stimulation(self):
        """
        Creates multiple current source as stimulation of the specified cluster/s.
        """
        if self.params.get('stim_clusters') is not None:
            stim_amp = self.params['stim_amp']  # amplitude of the stimulation current in pA
            stim_starts = self.params['stim_starts']  # list of stimulation start times
            stim_ends = self.params['stim_ends']  # list of stimulation end times
            amplitude_values = []
            amplitude_times = []
            for start, end in zip(stim_starts, stim_ends):
                amplitude_times.append(start + self.params['warmup'])
                amplitude_values.append(stim_amp)
                amplitude_times.append(end + self.params['warmup'])
                amplitude_values.append(0.)
            self.Currentsources = [nest.Create('step_current_generator')]


            for stim_cluster in self.params['stim_clusters']:
                nest.Connect(self.Currentsources[0], 
                                self.Populations[0][stim_cluster])
            nest.SetStatus(self.Currentsources[0],
                           {'amplitude_times': amplitude_times, 
                           'amplitude_values': amplitude_values,
                           'allow_offgrid_times':True})
        elif self.params['multi_stim_clusters'] is not None:
            print('stimulating multi stim ...')
            for stim_clusters,amplitudes,times in zip(self.params['multi_stim_clusters'],
                                                    self.params['multi_stim_amps'],
                                                    self.params['multi_stim_times']):
                self.Currentsources.append(nest.Create('step_current_generator'))
                nest.SetStatus(self.Currentsources[-1],{'amplitude_times':times[1:],
                                                'amplitude_values':amplitudes[1:],
                                                'allow_offgrid_times':True})
                stim_units = []
                for stim_cluster in stim_clusters:
                    nest.Connect(self.Currentsources[-1], 
                                self.Populations[0][stim_cluster])


    def create_recording_devices(self):
        """
        Creates a spike recorder connected to all neuron populations created by create_populations
        """
        self.RecordingDevices = [nest.Create("spike_detector")]
        nest.SetStatus(self.RecordingDevices[0], [
            {'to_file': False, 'withtime': True, 'withgid': True}])

        all_units = self.Populations[0][0]
        for E_pop in self.Populations[0][1:]:
            all_units += E_pop
        for I_pop in self.Populations[1]:
            all_units += I_pop
        nest.Connect(all_units, self.RecordingDevices[0], "all_to_all")  # Spikerecorder

        if self.params.get('record_voltage',False):
            self.VoltageRecorder = nest.Create("multimeter")
            recordables = self.params.get(
                'recordables', [str(r) for r in nest.GetStatus(
                    self.Populations[0][0], 'recordables')[0]])
            self.RecordingDevices.append(nest.Create('multimeter',
                                           params={'record_from': recordables,
                                                   'interval': self.params.get('recording_interval', self.params.get('dt', 0.1))})
                                             )
            if self.params.get("record_from", 'all'):
                record_units = []
                for E_pop in self.Populations[0]:
                    record_units += list(E_pop[:self.params.get("record_from", 'all')])
                for I_pop in self.Populations[1]:
                    record_units += list(I_pop[:self.params.get("record_from", 'all')])

            else:
                record_units = [u for u in all_units]

            nest.Connect( self.RecordingDevices[1], record_units)

    def get_voltage_recordings(self):
        if self.params.get('record_voltage',False):

            if self.params.get("record_from", 'all'):
                record_units = []
                for E_pop in self.Populations[0]:
                    record_units += list(E_pop[:self.params.get("record_from", 'all')])
                for I_pop in self.Populations[1]:
                    record_units += list(I_pop[:self.params.get("record_from", 'all')])

            else:
                all_units = self.Populations[0][0]
                for E_pop in self.Populations[0][1:]:
                    all_units += E_pop
                for I_pop in self.Populations[1]:
                    all_units += I_pop
                record_units = [u for u in all_units]

            print('extracting recordables')
            events = nest.GetStatus(self.RecordingDevices[1], 'events')[0]

            times = events['times']
            senders = events['senders']
            usenders = np.unique(senders)
            sender_ind_dict = {s: record_units.index(s) for s in usenders}
            sender_inds = [sender_ind_dict[s] for s in senders]

            utimes = np.unique(times)
            time_ind_dict = {t: i for i, t in enumerate(utimes)}
            time_inds = [time_ind_dict[t] for t in times]

            if self.params.get("record_from", 'all') == 'all':
                n_records = self.params.get('N_E', 0) + self.params.get('N_I', 0)
            else:
                n_records = self.params.get("record_from", 'all') * (len(self.Populations[0]) + len(self.Populations[1]))


            recordables = self.params.get(
                'recordables', [str(r) for r in nest.GetStatus(
                    self.Populations[0][0], 'recordables')[0]])

            results = {}
            for recordable in recordables:
                t0 = time.time()

                results[recordable] = np.zeros((n_records, len(utimes)))
                results[recordable][sender_inds, time_inds] = events[recordable]

                results[recordable] = results[recordable][:, utimes >= self.params['warmup']]

            utimes = utimes[utimes >= self.params['warmup']]
            utimes -= self.params['warmup']
            results['senders'] = np.array(record_units)
            results['times'] = utimes

            return results








    def setup_network(self):
        """
        Initializes NEST and creates the network in NEST, ready to be simulated.
        nest.Prepare is executed in this function.
        """
        self.setup_nest()
        self.create_populations()
        self.connect()
        self.create_recording_devices()
        self.create_stimulation()
        nest.Prepare()

    def simulate(self):
        """
        Simulates network for a period of warmup+simtime
        """
        if self.params['warmup'] + self.params['simtime'] <= 0.1:
            pass
        else:
            nest.Run(self.params['warmup'] + self.params['simtime'])

    def get_recordings(self):
        """
        Extracts spikes form the Spikerecorder connected to all populations created in create_populations.
        Cuts the warmup period away and sets time relative to end of warmup.
        Ids 1:N_E correspond to excitatory neurons, N_E+1:N_E+N_I correspond to inhibitory neurons.

        Returns:
            spiketimes (np.array): Row 0: spiketimes, Row 1: neuron ID.
        """
        events = nest.GetStatus(self.RecordingDevices[0], 'events')[0]
        # convert them to the format accepted by spiketools
        spiketimes = np.append(events['times'][None, :], events['senders'][None, :], axis=0)
        spiketimes[1] -= 1
        # remove the pre warmup spikes
        spiketimes = spiketimes[:, spiketimes[0] >= self.params['warmup']]
        spiketimes[0] -= self.params['warmup']
        return spiketimes

    def get_parameter(self):
        """
        Return:
            parameters (dict): Dictionary with all parameters for the simulation / network creation.
        """
        return self.params

    def create_and_simulate(self):
        """
        Creates the EI-clustered network and simulates it with the parameters supplied in the object creation.

        Returns:
            spiketimes (np.array):  Row 0: spiketimes, Row 1: neuron ID.
                                    Ids 1:N_E correspond to excitatory neurons,
                                    N_E+1:N_E+N_I correspond to inhibitory neurons.
        """
        self.setup_network()
        self.simulate()
        return self.get_recordings()


class ClusteredNetwork(ClusteredNetworkBase):
    """
    Adds to EI clustered network:
        Measurement of runtime (attribute Timing)
        Changeable ModelBuildPipeline (list of functions)
        Firing rate estimation of exc. and inh. neurons
        Functions to save connectivity and create connectivity from file
    """

    def __init__(self, defaultValues, parameters):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network. ClusteredNetwork objects
        measure the timing of the simulation and offer more functions than the base class.
        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        super().__init__(defaultValues, parameters)
        self.Timing = {}
        self.ModelBuildPipeline = [self.setup_nest, self.create_populations, self.create_stimulation,
                                   self.create_recording_devices, self.connect]

    def clean_network(self):
        super().clean_network()
        self.Timing = {}

    def setup_network(self):
        """
        Initializes NEST and creates the network in NEST, ready to be simulated.
        Functions saved in ModelBuildPipeline are executed.
        nest.Prepare is executed in this function.
        """
        startbuild = time.time()
        for func in self.ModelBuildPipeline:
            func()
        endbuild = time.time()
        endcompile = time.time()
        nest.Prepare()
        endLoad = time.time()
        self.Timing['Build'] = endbuild - startbuild
        self.Timing['Compile'] = 0.0
        self.Timing['Load'] = endLoad - endcompile

    def set_model_build_pipeline(self, Pipeline):
        """
        Sets the ModelBuildPipeline.
        Parameters:
            Pipeline (list of functions):   ordered list of functions executed to build the network model
        """
        self.ModelBuildPipeline = Pipeline

    def simulate(self):
        startsimulate = time.time()
        super().simulate()
        endsimulate = time.time()
        self.Timing['Sim'] = endsimulate - startsimulate

    def get_recordings(self):
        startPullSpikes = time.time()
        spiketimes = super().get_recordings()
        endPullSpikes = time.time()
        self.Timing['Download'] = endPullSpikes - startPullSpikes
        return spiketimes

    def set_I_x(self, I_XE, I_XI):
        for E_pop in self.Populations[0]:
            I_e_loc = E_pop.get('I_e')
            E_pop.set({'I_e': I_e_loc + I_XE})
        for I_pop in self.Populations[1]:
            I_e_loc = I_pop.get('I_e')
            I_pop.set({'I_e': I_e_loc + I_XI})

    def get_firing_rates(self, spiketimes=None):
        """
        Calculates the firing rates of all excitatory neurons and the firing rates of all inhibitory neurons
        created by self.create_populations. If spiketimes are not supplied, they get extracted.
        Parameters:
            spiketimes: (optional, np.array 2xT)   spiketimes of simulation
        Returns:
            (e_rate, i_rate) average firing rate of excitatory/inhibitory neurons (spikes/s)
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        e_count = spiketimes[:, spiketimes[1] < self.params['N_E']].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= self.params['N_E']].shape[1]
        e_rate = e_count / float(self.params['N_E']) / float(self.params['simtime']) * 1000.
        i_rate = i_count / float(self.params['N_I']) / float(self.params['simtime']) * 1000.
        return e_rate, i_rate

    def get_timing(self):
        """
        Gets Timing information of simulation.
        Returns:
            Dictionary with the timing information of the different simulation phases in seconds.
        """
        return self.Timing

    def get_simulation(self, PathSpikes=None, timeout=None):
        """
        Creates the network, simulates it and extracts the firing rates. If PathSpikes is supplied the spikes get saved
        to a pickle file. If a timeout is supplied, a timeout handler is created which stops the execution.
        Parameters:
            PathSpikes: (optional) Path of file for spiketimes
            timeout: (optional) Time of timeout in seconds
        Returns:
            Dictionary with firing rates, timing information (dict) and parameters (dict)
        """
        if timeout is not None:
            # Change the behavior of SIGALRM
            signal.signal(signal.SIGALRM, GeneralHelper.timeout_handler)
            signal.alarm(timeout)
            # This try/except loop ensures that 
            #   you'll catch TimeoutException when it's sent.
        try:
            self.setup_network()
            self.simulate()
            spiketimes = self.get_recordings()
            e_rate, i_rate = self.get_firing_rates(spiketimes)

            if PathSpikes is not None:
                with open(PathSpikes, 'wb') as outfile:
                    pickle.dump(spiketimes, outfile)

            nest.Cleanup()
            return {'e_rate': e_rate, 'i_rate': i_rate, 
                    'Timing': self.get_timing(), 
                    'params': self.get_parameter(),
                    'spiketimes': spiketimes}

        except GeneralHelper.TimeoutException:
            print("Aborted - Timeout")
            return {'e_rate': -1, 'i_rate': -1, 'Timing': self.get_timing(), 
                    'params': self.get_parameter(),
                    'spiketimes': [[], []]}

    def connect_from_file(self, Filename):
        """
        Connects the network from saved connectivity.
        Parameters:
              Filename (str or filedescriptor): Path to file with saved connectivity to be loaded.
        """

        assert nest.NumProcesses() == 1, "Cannot load MPI parallel"
        with open(Filename, "rb") as f:
            network = pickle.load(f)
        assert network["n_vp"] == nest.total_num_virtual_procs, "N_VP must match"

        ###############################################################################
        # Reconstruct connectivity
        # EE
        nest.CopyModel("static_synapse", "EE_plus", {"delay": self.params['delay']})
        nest.CopyModel("static_synapse", "EE_minus", {"delay": self.params['delay']})

        nest.Connect(network["EE_plus"].source.values, network["EE_plus"].target.values,
                     "one_to_one",
                     {"synapse_model": "EE_plus", "weight": network["EE_plus"].weight.values})
        nest.Connect(network["EE_minus"].source.values, network["EE_minus"].target.values,
                     "one_to_one",
                     {"synapse_model": "EE_minus", "weight": network["EE_minus"].weight.values})

        # EI
        nest.CopyModel("static_synapse", "EI_plus", {"delay": self.params['delay']})
        nest.CopyModel("static_synapse", "EI_minus", {"delay": self.params['delay']})

        nest.Connect(network["EI_plus"].source.values, network["EI_plus"].target.values,
                     "one_to_one",
                     {"synapse_model": "EI_plus", "weight": network["EI_plus"].weight.values})
        nest.Connect(network["EI_minus"].source.values, network["EI_minus"].target.values,
                     "one_to_one",
                     {"synapse_model": "EI_minus", "weight": network["EI_minus"].weight.values})

        # IE
        nest.CopyModel("static_synapse", "IE_plus", {"delay": self.params['delay']})
        nest.CopyModel("static_synapse", "IE_minus", {"delay": self.params['delay']})

        nest.Connect(network["IE_plus"].source.values, network["IE_plus"].target.values,
                     "one_to_one",
                     {"synapse_model": "IE_plus", "weight": network["IE_plus"].weight.values})
        nest.Connect(network["IE_minus"].source.values, network["IE_minus"].target.values,
                     "one_to_one",
                     {"synapse_model": "IE_minus", "weight": network["IE_minus"].weight.values})

        # II
        nest.CopyModel("static_synapse", "II_plus", {"delay": self.params['delay']})
        nest.CopyModel("static_synapse", "II_minus", {"delay": self.params['delay']})

        nest.Connect(network["II_plus"].source.values, network["II_plus"].target.values,
                     "one_to_one",
                     {"synapse_model": "II_plus", "weight": network["II_plus"].weight.values})
        nest.Connect(network["II_minus"].source.values, network["II_minus"].target.values,
                     "one_to_one",
                     {"synapse_model": "II_minus", "weight": network["II_minus"].weight.values})

    @staticmethod
    def save_conn_to_file(Filename):
        """
        Saves the network connectivity of the E and I populations to file.
        Parameters:
              Filename (str or filedescriptor): Filepath to store connectivity.
        """
        assert nest.NumProcesses() == 1, "Cannot dump MPI parallel"
        import pickle
        network = {"n_vp": nest.total_num_virtual_procs,
                   "EE_plus": nest.GetConnections(synapse_model="EE_plus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "EE_minus": nest.GetConnections(synapse_model="EE_minus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "EI_plus": nest.GetConnections(synapse_model="EI_plus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "EI_minus": nest.GetConnections(synapse_model="EI_minus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "IE_plus": nest.GetConnections(synapse_model="IE_plus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "IE_minus": nest.GetConnections(synapse_model="IE_minus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "II_plus": nest.GetConnections(synapse_model="II_plus").get(
                       ("source", "target", "weight"), output="pandas"),
                   "II_minus": nest.GetConnections(synapse_model="II_minus").get(
                       ("source", "target", "weight"), output="pandas")}

        with open(Filename, "wb") as f:
            pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # np.random.seed(42)
    sys.path.append("..")
    import default
    import matplotlib.pyplot as plt

    EI_cluster = ClusteredNetworkBase(default, 
                        {'Q':20,'n_jobs': 4, 'warmup': 500, 'simtime': 1200, 'multi_stim_clusters': [[1], [2], [3], [4], [5], [6]],
                        'multi_stim_amps': [[0.1, 0.] for i in range(6)],
                        'multi_stim_times': [[100., 200.] for i in range(6)],
                        })
    spikes = EI_cluster.create_and_simulate()
