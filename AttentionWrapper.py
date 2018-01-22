import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, \
    AttentionWrapperState, _compute_attention


class MyBahdanauAttention(_BaseAttentionMechanism):
  def __init__(self, num_units, memory, memory_sequence_length=None, normalize=False, score_mask_value=float("-inf"),
               name="BahdanauAttention"):

    wrapped_probability_fn = lambda score, _: tf.nn.softmax(score)
    super(MyBahdanauAttention, self).__init__(
        query_layer=tf.layers.Dense(
            num_units, name="query_layer", use_bias=False),
        memory_layer=tf.layers.Dense(
            num_units, name="memory_layer", use_bias=False),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._name = name

  def __call__(self, query, previous_alignments):
    with tf.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      num_units = self._num_units
      processed_query = array_ops.expand_dims(processed_query, 1)
      v = tf.get_variable("attention_v", [num_units], dtype=tf.float32)
      score = tf.reduce_sum(v * tf.tanh(self._keys + processed_query), [2])
    alignments = self._probability_fn(score, previous_alignments)
    return alignments

class MyAttentionWrapper(rnn_cell_impl.RNNCell):
    def __init__(self, cell, attention_mechanism,  attention_layer_size=None, alignment_history=False,
                 output_attention=True, name=None):
        super(MyAttentionWrapper, self).__init__(name=name)
        self._is_multi = False
        attention_mechanisms = (attention_mechanism,)
        cell_input_fn = (lambda inputs, attention: array_ops.concat([inputs, attention], -1))

        attention_layer_sizes = tuple(attention_layer_size if isinstance(attention_layer_size, (list, tuple))
                                      else (attention_layer_size,))

        self._attention_layers = tuple(tf.layers.Dense(attention_layer_size, name="attention_layer", use_bias=False)
                                       for attention_layer_size in attention_layer_sizes)
        self._attention_layer_size = sum(attention_layer_sizes)

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        self._initial_cell_state = None


    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size,
                                attention_mechanism.batch_size,
                                message=error_message)
                for attention_mechanism in self._attention_mechanisms]


    def _item_or_tuple(self, seq):
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]


    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size


    @property
    def state_size(self):
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                () for _ in self._attention_mechanisms))  # sometimes a TensorArray


    def zero_state(self, batch_size, dtype):
        cell_state = self._cell.zero_state(batch_size, dtype)
        cell_state = tf.contrib.framework.nest.map_structure(lambda s: array_ops.identity(s, name="checked_cell_state"), cell_state)

        return AttentionWrapperState(
            cell_state=cell_state,
            time=array_ops.zeros([], dtype=tf.int32),
            attention=tf.zeros(shape=[batch_size, self._attention_layer_size], dtype = dtype),
            alignments=self._item_or_tuple(
                attention_mechanism.initial_alignments(batch_size, dtype)
                for attention_mechanism in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                tf.TensorArray(dtype=dtype, size=0,
                                             dynamic_size=True)
                if self._alignment_history else ()
                for _ in self._attention_mechanisms))


    def call(self, inputs, state):
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_output = array_ops.identity(cell_output, name="checked_cell_output")

        previous_alignments = [state.alignments]
        previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments = _compute_attention(
                attention_mechanism, cell_output, previous_alignments[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = array_ops.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
