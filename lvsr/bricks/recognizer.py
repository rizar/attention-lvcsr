import numpy
import theano
import logging
from theano import tensor

from blocks.bricks import (
    Bias, Identity, Initializable, MLP, Tanh)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.recurrent import (
    BaseRecurrent, RecurrentStack)
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout,
    SoftmaxEmitter, LookupFeedback)
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import OUTPUT
from blocks.search import BeamSearch
from blocks.serialization import load_parameter_values

from lvsr.bricks import (
    Encoder, OneOfNFeedback, InitializableSequence, RewardRegressionEmitter)
from lvsr.bricks.attention import SequenceContentAndConvAttention
from lvsr.bricks.language_models import (
    LanguageModel, LMEmitter, ShallowFusionReadout)
from lvsr.utils import global_push_initialization_config, SpeechModel

logger = logging.getLogger(__name__)


class Bottom(Initializable):
    """
    A bottom class that mergers possibly many input sources into one
    sequence.

    The bottom is responsible for allocating variables for single and
    multiple sequences in a batch.

    In speech recognition this will typically be the identity transformation
    ro a small MLP.

    Attributes
    ----------
    vector_input_sources : list of str
    discrete_input_sources : list of str

    Parameters
    ----------
    input_dims : dict
        Maps input source to their dimensions, only for vector sources.
    input_num_chars : dict
        Maps input source to their range of values, only for discrete sources.

    """
    vector_input_sources = []
    discrete_input_sources = []

    def __init__(self, input_dims, input_num_chars, **kwargs):
        super(Bottom, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.input_num_chars = input_num_chars


class LookupBottom(Bottom):
    discrete_input_sources = ['inputs']

    def __init__(self, dim, **kwargs):
        super(LookupBottom, self).__init__(**kwargs)
        self.dim = dim

        self.mask = tensor.matrix('inputs_mask')
        self.batch_inputs = {
            'inputs': tensor.lmatrix('inputs')}
        self.single_inputs = {
            'inputs': tensor.lvector('inputs')}

        self.children = [LookupTable(self.input_num_chars['inputs'], self.dim)]

    @application(inputs=['inputs'], outputs=['outputs'])
    def apply(self, inputs):
        return self.children[0].apply(inputs)

    def batch_size(self, inputs):
        return inputs.shape[1]

    def num_time_steps(self, inputs):
        return inputs.shape[0]

    def single_to_batch_inputs(self, inputs):
        # Note: this code supports many inputs, which are all sequences
        inputs = {n: v[:, None, :] if v.ndim == 2 else v[:, None]
                  for (n, v) in inputs.items()}
        inputs_mask = tensor.ones((self.num_time_steps(**inputs),
                                   self.batch_size(**inputs)))
        return inputs, inputs_mask

    def get_dim(self, name):
        if name == 'outputs':
            return self.dim
        return super(LookupBottom, self).get_dim(name)


class SpeechBottom(Bottom):
    """
    A Bottom specialized for speech recognition that accets only one input
    - the recordings.
    """
    input_sources = 'recordings'

    def __init__(self, activation, dims=None, **kwargs):
        super(SpeechBottom, self).__init__(**kwargs)
        self.num_features = self.input_source_dims['recordings']

        if activation is None:
            activation = Tanh()

        if dims:
            child = MLP([activation] * len(dims),
                        [self.num_features] + dims,
                        name="bottom")
            self.output_dim = child.output_dim
        else:
            child = Identity(name='bottom')
            self.output_dim = self.num_features
        self.children.append(child)

        self.mask = tensor.matrix('recordings_mask')
        self.batch_inputs = {
            'recordings': tensor.tensor3('recordings')}
        self.single_inputs = {
            'recordings': tensor.matrix('recordings')}

    @application(inputs=['recordings'], outputs=['outputs'])
    def apply(self, recordings):
        return self.children[0].apply(recordings)

    def batch_size(self, recordings):
        return recordings.shape[1]

    def num_time_steps(self, recordings):
        return recordings.shape[0]

    def single_to_batch_inputs(self, inputs):
        # Note: this code supports many inputs, which are all sequences
        inputs = {n: v[:, None, :] if v.ndim == 2 else v[:, None]
                  for (n, v) in inputs.items()}
        inputs_mask = tensor.ones((self.num_time_steps(**inputs),
                                   self.batch_size(**inputs)))
        return inputs, inputs_mask

    def get_dim(self, name):
        if name == 'outputs':
            return self.output_dim
        return super(SpeechBottom, self).get_dim(name)


class SpeechRecognizer(Initializable):
    """Encapsulate all reusable logic.

    This class plays a few roles: (a) it's a top brick that knows
    how to combine bottom, bidirectional and recognizer network, (b)
    it has the inputs variables and can build whole computation graphs
    starting with them (c) it hides compilation of Theano functions
    and initialization of beam search. I find it simpler to have it all
    in one place for research code.

    Parameters
    ----------
    All defining the structure and the dimensions of the model. Typically
    receives everything from the "net" section of the config.

    """

    def __init__(self,
                 input_dims,
                 input_num_chars,
                 eos_label,
                 num_phonemes,
                 dim_dec, dims_bidir,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 criterion,
                 bottom,
                 lm=None, character_map=None,
                 bidir=True,
                 subsample=None,
                 dims_top=None,
                 prior=None, conv_n=None,
                 post_merge_activation=None,
                 post_merge_dims=None,
                 dim_matcher=None,
                 embed_outputs=True,
                 dim_output_embedding=None,
                 dec_stack=1,
                 conv_num_filters=1,
                 data_prepend_eos=True,
                 # softmax is the default set in SequenceContentAndConvAttention
                 energy_normalizer=None,
                 # for speech this is the approximate phoneme duration in frames
                 max_decoded_length_scale=3,
                 **kwargs):

        if post_merge_activation is None:
            post_merge_activation = Tanh()
        super(SpeechRecognizer, self).__init__(**kwargs)
        self.eos_label = eos_label
        self.data_prepend_eos = data_prepend_eos

        self.rec_weights_init = None
        self.initial_states_init = None

        self.enc_transition = enc_transition
        self.dec_transition = dec_transition
        self.dec_stack = dec_stack

        self.criterion = criterion

        self.max_decoded_length_scale = max_decoded_length_scale

        post_merge_activation = post_merge_activation

        if dim_matcher is None:
            dim_matcher = dim_dec

        # The bottom part, before BiRNN
        bottom_class = bottom.pop('bottom_class')
        bottom = bottom_class(
            input_dims=input_dims, input_num_chars=input_num_chars,
            name='bottom',
            **bottom)

        # BiRNN
        if not subsample:
            subsample = [1] * len(dims_bidir)
        encoder = Encoder(self.enc_transition, dims_bidir,
                          bottom.get_dim(bottom.apply.outputs[0]),
                          subsample, bidir=bidir)
        dim_encoded = encoder.get_dim(encoder.apply.outputs[0])

        # The top part, on top of BiRNN but before the attention
        if dims_top:
            top = MLP([Tanh()],
                      [dim_encoded] + dims_top + [dim_encoded], name="top")
        else:
            top = Identity(name='top')

        if dec_stack == 1:
            transition = self.dec_transition(
                dim=dim_dec, activation=Tanh(), name="transition")
        else:
            transitions = [self.dec_transition(dim=dim_dec,
                                               activation=Tanh(),
                                               name="transition_{}".format(trans_level))
                           for trans_level in xrange(dec_stack)]
            transition = RecurrentStack(transitions=transitions,
                                        skip_connections=True)
        # Choose attention mechanism according to the configuration
        if attention_type == "content":
            attention = SequenceContentAttention(
                state_names=transition.apply.states,
                attended_dim=dim_encoded, match_dim=dim_matcher,
                name="cont_att")
        elif attention_type == "content_and_conv":
            attention = SequenceContentAndConvAttention(
                state_names=transition.apply.states,
                conv_n=conv_n,
                conv_num_filters=conv_num_filters,
                attended_dim=dim_encoded, match_dim=dim_matcher,
                prior=prior,
                energy_normalizer=energy_normalizer,
                name="conv_att")
        else:
            raise ValueError("Unknown attention type {}"
                             .format(attention_type))
        if embed_outputs:
            feedback = LookupFeedback(num_phonemes + 1,
                                      dim_dec if
                                      dim_output_embedding is None
                                      else dim_output_embedding)
        else:
            feedback = OneOfNFeedback(num_phonemes + 1)
        if criterion['name'] == 'log_likelihood':
            emitter = SoftmaxEmitter(initial_output=num_phonemes, name="emitter")
            if lm:
                # In case we use LM it is Readout that is responsible
                # for normalization.
                emitter = LMEmitter()
        elif criterion['name'].startswith('mse'):
            emitter = RewardRegressionEmitter(
                criterion['name'], eos_label, num_phonemes,
                criterion.get('min_reward', -1.0),
                name="emitter")
        else:
            raise ValueError("Unknown criterion {}".format(criterion['name']))
        readout_config = dict(
            readout_dim=num_phonemes,
            source_names=(transition.apply.states if use_states_for_readout else [])
                         + [attention.take_glimpses.outputs[0]],
            emitter=emitter,
            feedback_brick=feedback,
            name="readout")
        if post_merge_dims:
            readout_config['merged_dim'] = post_merge_dims[0]
            readout_config['post_merge'] = InitializableSequence([
                Bias(post_merge_dims[0]).apply,
                post_merge_activation.apply,
                MLP([post_merge_activation] * (len(post_merge_dims) - 1) + [Identity()],
                    # MLP was designed to support Maxout is activation
                    # (because Maxout in a way is not one). However
                    # a single layer Maxout network works with the trick below.
                    # For deeper Maxout network one has to use the
                    # Sequence brick.
                    [d//getattr(post_merge_activation, 'num_pieces', 1)
                     for d in post_merge_dims] + [num_phonemes]).apply,
            ],
                name='post_merge')
        readout = Readout(**readout_config)

        language_model = None
        if lm and lm.get('path'):
            lm_weight = lm.pop('weight', 0.0)
            normalize_am_weights = lm.pop('normalize_am_weights', True)
            normalize_lm_weights = lm.pop('normalize_lm_weights', False)
            normalize_tot_weights = lm.pop('normalize_tot_weights', False)
            am_beta = lm.pop('am_beta', 1.0)
            if normalize_am_weights + normalize_lm_weights + normalize_tot_weights < 1:
                logger.warn("Beam search is prone to fail with no log-prob normalization")
            language_model = LanguageModel(nn_char_map=character_map, **lm)
            readout = ShallowFusionReadout(lm_costs_name='lm_add',
                                           lm_weight=lm_weight,
                                           normalize_am_weights=normalize_am_weights,
                                           normalize_lm_weights=normalize_lm_weights,
                                           normalize_tot_weights=normalize_tot_weights,
                                           am_beta=am_beta,
                                           **readout_config)

        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            language_model=language_model,
            name="generator")

        # Remember child bricks
        self.encoder = encoder
        self.bottom = bottom
        self.top = top
        self.generator = generator
        self.children = [encoder, top, bottom, generator]

        # Create input variables
        self.inputs = self.bottom.batch_inputs
        self.inputs_mask = self.bottom.mask

        self.labels = tensor.lmatrix('labels')
        self.labels_mask = tensor.matrix("labels_mask")

        self.single_inputs = self.bottom.single_inputs
        self.single_labels = tensor.lvector('labels')
        self.n_steps = tensor.lscalar('n_steps')

    def push_initialization_config(self):
        super(SpeechRecognizer, self).push_initialization_config()
        if self.rec_weights_init:
            rec_weights_config = {'weights_init': self.rec_weights_init,
                                  'recurrent_weights_init': self.rec_weights_init}
            global_push_initialization_config(self,
                                              rec_weights_config,
                                              BaseRecurrent)
        if self.initial_states_init:
            global_push_initialization_config(self,
                                              {'initial_states_init': self.initial_states_init})

    @application
    def cost(self, **kwargs):
        # pop inputs we know about
        inputs_mask = kwargs.pop('inputs_mask')
        labels = kwargs.pop('labels')
        labels_mask = kwargs.pop('labels_mask')

        # the rest is for bottom
        bottom_processed = self.bottom.apply(**kwargs)
        encoded, encoded_mask = self.encoder.apply(
            input_=bottom_processed,
            mask=inputs_mask)
        encoded = self.top.apply(encoded)
        return self.generator.cost_matrix(
            labels, labels_mask,
            attended=encoded, attended_mask=encoded_mask)

    @application
    def generate(self, **kwargs):
        inputs_mask = kwargs.pop('inputs_mask')
        n_steps = kwargs.pop('n_steps')

        encoded, encoded_mask = self.encoder.apply(
            input_=self.bottom.apply(**kwargs),
            mask=inputs_mask)
        encoded = self.top.apply(encoded)
        return self.generator.generate(
            n_steps=n_steps if n_steps is not None else self.n_steps,
            batch_size=encoded.shape[1],
            attended=encoded,
            attended_mask=encoded_mask,
            as_dict=True)

    def load_params(self, path):
        generated = self.get_generate_graph()
        param_values = load_parameter_values(path)
        SpeechModel(generated['outputs']).set_parameter_values(param_values)

    def get_generate_graph(self, use_mask=True, n_steps=None):
        inputs_mask = None
        if use_mask:
            inputs_mask = self.inputs_mask
        bottom_inputs = self.inputs
        return self.generate(n_steps=n_steps,
                             inputs_mask=inputs_mask,
                             **bottom_inputs)

    def get_cost_graph(self, batch=True,
                       prediction=None, prediction_mask=None):

        if batch:
            inputs = self.inputs
            inputs_mask = self.inputs_mask
            groundtruth = self.labels
            groundtruth_mask = self.labels_mask
        else:
            inputs, inputs_mask = self.bottom.single_to_batch_inputs(
                self.single_inputs)
            groundtruth = self.single_labels[:, None]
            groundtruth_mask = None

        if not prediction:
            prediction = groundtruth
        if not prediction_mask:
            prediction_mask = groundtruth_mask

        cost = self.cost(inputs_mask=inputs_mask,
                         labels=prediction,
                         labels_mask=prediction_mask,
                         **inputs)
        cost_cg = ComputationGraph(cost)
        if self.criterion['name'].startswith("mse"):
            placeholder, = VariableFilter(theano_name='groundtruth')(cost_cg)
            cost_cg = cost_cg.replace({placeholder: groundtruth})
        return cost_cg

    def analyze(self, inputs, groundtruth, prediction=None):
        """Compute cost and aligment."""

        input_values_dict = dict(inputs)
        input_values_dict['groundtruth'] = groundtruth
        if prediction is not None:
            input_values_dict['prediction'] = prediction
        if not hasattr(self, "_analyze"):
            input_variables = list(self.single_inputs.values())
            input_variables.append(self.single_labels.copy(name='groundtruth'))

            prediction_variable = tensor.lvector('prediction')
            if prediction is not None:
                input_variables.append(prediction_variable)
                cg = self.get_cost_graph(
                    batch=False, prediction=prediction_variable[:, None])
            else:
                cg = self.get_cost_graph(batch=False)
            cost = cg.outputs[0]

            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)

            energies = VariableFilter(
                bricks=[self.generator], name="energies")(cg)
            energies_output = [energies[0][:, 0, :] if energies
                               else tensor.zeros_like(weights)]

            states, = VariableFilter(
                applications=[self.encoder.apply], roles=[OUTPUT],
                name="encoded")(cg)

            ctc_matrix_output = []
            # Temporarily disabled for compatibility with LM code
            # if len(self.generator.readout.source_names) == 1:
            #    ctc_matrix_output = [
            #        self.generator.readout.readout(weighted_averages=states)[:, 0, :]]

            self._analyze = theano.function(
                input_variables,
                [cost[:, 0], weights[:, 0, :]] + energies_output + ctc_matrix_output,
                on_unused_input='warn')
        return self._analyze(**input_values_dict)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
        if hasattr(self, '_beam_search') and self.beam_size == beam_size:
            # Only recompile if the user wants a different beam size
            return
        self.beam_size = beam_size
        generated = self.get_generate_graph(use_mask=False, n_steps=3)
        cg = ComputationGraph(generated.values())
        samples, = VariableFilter(
            applications=[self.generator.generate], name="outputs")(cg)
        self._beam_search = BeamSearch(beam_size, samples)
        self._beam_search.compile()

    def beam_search(self, inputs, **kwargs):
        # When a recognizer is unpickled, self.beam_size is available
        # but beam search has to be recompiled.

        self.init_beam_search(self.beam_size)
        inputs = dict(inputs)
        max_length = int(self.bottom.num_time_steps(**inputs) /
                         self.max_decoded_length_scale)
        search_inputs = {}
        for var in self.inputs.values():
            search_inputs[var] = inputs.pop(var.name)[:, numpy.newaxis, ...]
        if inputs:
            raise Exception(
                'Unknown inputs passed to beam search: {}'.format(
                    inputs.keys()))
        outputs, search_costs = self._beam_search.search(
            search_inputs, self.eos_label,
            max_length,
            ignore_first_eol=self.data_prepend_eos,
            **kwargs)
        return outputs, search_costs

    def init_generate(self):
        generated = self.get_generate_graph(use_mask=False)
        inputs = [v.copy(name=n) for (n, v) in self.inputs.items()]
        inputs.append(self.n_steps.copy(name='n_steps'))
        self._do_generate = theano.function(inputs, generated)

    def sample(self, inputs, n_steps=None):
        if not hasattr(self, '_do_generate'):
            self.init_generate()
        batch, unused_mask = self.bottom.single_to_batch_inputs(inputs)
        batch['n_steps'] = n_steps if n_steps is not None \
            else int(self.bottom.num_time_steps(**batch) /
                     self.max_decoded_length_scale)
        return self._do_generate(**batch)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # To use bricks used on a GPU first on a CPU later
        try:
            emitter = self.generator.readout.emitter
            del emitter._theano_rng
        except:
            pass
