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
    def __init__(self, recordings_source, labels_source, eos_label,
                 num_features, num_phonemes,
                 dim_dec, dims_bidir, dims_bottom,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 criterion,
                 lm=None, character_map=None,
                 subsample=None,
                 dims_top=None,
                 prior=None, conv_n=None,
                 bottom_activation=None,
                 post_merge_activation=None,
                 post_merge_dims=None,
                 dim_matcher=None,
                 embed_outputs=True,
                 dec_stack=1,
                 conv_num_filters=1,
                 data_prepend_eos=True,
                 energy_normalizer=None,  # softmax is the default set in SequenceContentAndConvAttention
                 **kwargs):
        if bottom_activation is None:
            bottom_activation = Tanh()
        if post_merge_activation is None:
            post_merge_activation = Tanh()
        super(SpeechRecognizer, self).__init__(**kwargs)
        self.recordings_source = recordings_source
        self.labels_source = labels_source
        self.eos_label = eos_label
        self.data_prepend_eos = data_prepend_eos

        self.rec_weights_init = None
        self.initial_states_init = None

        self.enc_transition = enc_transition
        self.dec_transition = dec_transition
        self.dec_stack = dec_stack

        self.criterion = criterion

        bottom_activation = bottom_activation
        post_merge_activation = post_merge_activation

        if dim_matcher is None:
            dim_matcher = dim_dec

        # The bottom part, before BiRNN
        if dims_bottom:
            bottom = MLP([bottom_activation] * len(dims_bottom),
                         [num_features] + dims_bottom,
                         name="bottom")
        else:
            bottom = Identity(name='bottom')

        # BiRNN
        if not subsample:
            subsample = [1] * len(dims_bidir)
        encoder = Encoder(self.enc_transition, dims_bidir,
                          dims_bottom[-1] if len(dims_bottom) else num_features,
                          subsample)

        # The top part, on top of BiRNN but before the attention
        if dims_top:
            top = MLP([Tanh()],
                      [2 * dims_bidir[-1]] + dims_top + [2 * dims_bidir[-1]], name="top")
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
                attended_dim=2 * dims_bidir[-1], match_dim=dim_matcher,
                name="cont_att")
        elif attention_type == "content_and_conv":
            attention = SequenceContentAndConvAttention(
                state_names=transition.apply.states,
                conv_n=conv_n,
                conv_num_filters=conv_num_filters,
                attended_dim=2 * dims_bidir[-1], match_dim=dim_matcher,
                prior=prior,
                energy_normalizer=energy_normalizer,
                name="conv_att")
        else:
            raise ValueError("Unknown attention type {}"
                             .format(attention_type))
        if embed_outputs:
            feedback = LookupFeedback(num_phonemes + 1, dim_dec)
        else:
            feedback = OneOfNFeedback(num_phonemes + 1)
        if lm:
            # In case we use LM it is Readout that is responsible
            # for normalization.
            if criterion['name'] != 'log_likelihood':
                raise ValueError('LM integration only for log-likelihood')
            emitter = LMEmitter()
        if criterion['name'] == 'log_likelihood':
            emitter = SoftmaxEmitter(initial_output=num_phonemes, name="emitter")
        elif criterion['name'].startswith('mse'):
            emitter = RewardRegressionEmitter(
                criterion['name'], eos_label, num_phonemes,
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
        if lm:
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
        self.recordings = tensor.tensor3(self.recordings_source)
        self.recordings_mask = tensor.matrix(self.recordings_source + "_mask")
        self.labels = tensor.lmatrix(self.labels_source)
        self.labels_mask = tensor.matrix(self.labels_source + "_mask")
        self.batch_inputs = [self.recordings, self.recordings_source,
                             self.labels, self.labels_mask]
        self.single_recording = tensor.matrix(self.recordings_source)
        self.single_transcription = tensor.lvector(self.labels_source)
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

    @application(inputs=['recordings', 'recordings_mask',
                         'labels', 'labels_mask'])
    def cost(self, recordings, recordings_mask, labels, labels_mask):
        bottom_processed = self.bottom.apply(recordings)
        encoded, encoded_mask = self.encoder.apply(
            input_=bottom_processed,
            mask=recordings_mask)
        encoded = self.top.apply(encoded)
        return self.generator.cost_matrix(
            labels, labels_mask,
            attended=encoded, attended_mask=encoded_mask)

    @application
    def generate(self, recordings, recordings_mask, n_steps=None):
        encoded, encoded_mask = self.encoder.apply(
            input_=self.bottom.apply(recordings),
            mask=recordings_mask)
        encoded = self.top.apply(encoded)
        return self.generator.generate(
            n_steps=n_steps if n_steps is not None else self.n_steps,
            batch_size=recordings.shape[1],
            attended=encoded,
            attended_mask=encoded_mask,
            as_dict=True)

    def load_params(self, path):
        generated = self.get_generate_graph()
        param_values = load_parameter_values(path)
        SpeechModel(generated['outputs']).set_parameter_values(param_values)

    def get_generate_graph(self, use_mask=True, n_steps=None):
        return self.generate(self.recordings, self.recordings_mask if use_mask else None,
                             n_steps)

    def get_cost_graph(self, batch=True,
                       prediction=None, prediction_mask=None):
        if batch:
            recordings = self.recordings
            recordings_mask = self.recordings_mask
            groundtruth = self.labels
            groundtruth_mask = self.labels_mask
        else:
            recordings = self.single_recording[:, None, :]
            recordings_mask = tensor.ones_like(recordings[:, :, 0])
            groundtruth = self.single_transcription[:, None]
            groundtruth_mask = None
        if not prediction:
            prediction = groundtruth
        if not prediction_mask:
            prediction_mask = groundtruth_mask
        cost = self.cost(recordings, recordings_mask,
                         prediction, prediction_mask)
        cost_cg = ComputationGraph(cost)
        if self.criterion['name'].startswith("mse"):
            placeholder, = VariableFilter(theano_name='groundtruth')(cost_cg)
            cost_cg = cost_cg.replace({placeholder: groundtruth})
        return cost_cg

    def analyze(self, recording, groundtruth, prediction=None):
        """Compute cost and aligment."""
        input_values = [recording, groundtruth]
        if prediction is not None:
            input_values.append(prediction)
        if not hasattr(self, "_analyze"):
            input_variables = [self.single_recording, self.single_transcription]
            prediction_variable = tensor.lvector('prediction')
            if prediction is not None:
                input_variables.append(prediction_variable)
                cg = self.get_cost_graph(
                    batch=False, prediction=prediction_variable[:, None])
            else:
                cg = self.get_cost_graph(batch=False)
            cost = cg.outputs[0]
            energies = VariableFilter(
                bricks=[self.generator], name="energies")(cg)
            energies_output = [energies[0][:, 0, :] if energies
                               else tensor.zeros((self.single_transcription.shape[0],
                                                  self.single_recording.shape[0]))]
            states, = VariableFilter(
                applications=[self.encoder.apply], roles=[OUTPUT],
                name="encoded")(cg)
            ctc_matrix_output = []
            # Temporarily disabled for compatibility with LM code
            # if len(self.generator.readout.source_names) == 1:
            #    ctc_matrix_output = [
            #        self.generator.readout.readout(weighted_averages=states)[:, 0, :]]
            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)
            self._analyze = theano.function(
                input_variables,
                [cost[:, 0], weights[:, 0, :]] + energies_output + ctc_matrix_output)
        return self._analyze(*input_values)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
        self.beam_size = beam_size
        generated = self.get_generate_graph(use_mask=False, n_steps=3)
        cg = ComputationGraph(generated.values())
        samples, = VariableFilter(
            applications=[self.generator.generate], name="outputs")(cg)
        self._beam_search = BeamSearch(beam_size, samples)
        self._beam_search.compile()

    def beam_search(self, recording, **kwargs):
        if not hasattr(self, '_beam_search'):
            self.init_beam_search(self.beam_size)
        input_ = recording[:,numpy.newaxis,:]
        outputs, search_costs = self._beam_search.search(
            {self.recordings: input_}, self.eos_label, input_.shape[0] / 3,
            ignore_first_eol=self.data_prepend_eos,
            **kwargs)
        return outputs, search_costs

    def init_generate(self):
        generated = self.get_generate_graph(use_mask=False)
        self._do_generate = theano.function(
            [self.recordings, self.n_steps], generated)

    def sample(self, recording, n_steps=None):
        if not hasattr(self, '_do_generate'):
            self.init_generate()
        batch = recording[:, None, :]
        return self._do_generate(
            batch, n_steps if n_steps is not None else recording.shape[0] / 3)

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
