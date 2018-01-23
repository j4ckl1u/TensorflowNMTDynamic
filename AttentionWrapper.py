import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, \
    AttentionWrapperState, _compute_attention


class MyAttention(_BaseAttentionMechanism):

    def __init__(self, num_units, memory, memory_sequence_length=None, scope="MyAttention"):
        self._name = scope + "_MyAttention"
        self._num_units = num_units

        with tf.variable_scope(scope):
            query_layer = tf.layers.Dense(num_units, name="query_layer", use_bias=False)
            memory_layer = tf.layers.Dense(num_units, name="memory_layer", use_bias=False)
            self.v = tf.get_variable("attention_v", [num_units], dtype=tf.float32)
        wrapped_probability_fn = lambda score, _: tf.nn.softmax(score)

        super(MyAttention, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=float("-inf"),
            name=self._name)

    def __call__(self, query, previous_alignments):

        processed_query = self.query_layer(query)
        processed_query = tf.expand_dims(processed_query, 1)
        score = tf.reduce_sum(self.v * tf.tanh(self._keys + processed_query), [2])
        alignments = self._probability_fn(score, previous_alignments)
        return alignments


class MyAttentionWrapper(RNNCell):
    def __init__(self, cell, attention_mechanism,  attention_layer_size=None, alignment_history=False, scope="MyAttentionWrapper"):

        super(MyAttentionWrapper, self).__init__(name=scope+"_AttentionWrapper")
        with tf.variable_scope(scope):
            self.attention_layer = tf.layers.Dense(attention_layer_size, name="attention_layer", use_bias=False)

        self.attention_layer_size = attention_layer_size
        self.cell = cell
        self.attention_mechanism = attention_mechanism
        self.cell_input_fn = (lambda inputs, attention: tf.concat([inputs, attention], -1))
        self.use_alignment_history = alignment_history

    @property
    def output_size(self):
        return self.attention_layer_size

    @property
    def state_size(self):
        return AttentionWrapperState(
            cell_state=self.cell.state_size,
            time=tf.TensorShape([]),
            attention=self.attention_layer_size,
            alignments=self.attention_mechanism.alignments_size,
            alignment_history=())  # sometimes a TensorArray


    def zero_state(self, batch_size, dtype):
        cell_state = self.cell.zero_state(batch_size, dtype)
        cell_state = tf.contrib.framework.nest.map_structure(lambda s: tf.identity(s, name="checked_cell_state"), cell_state)

        return AttentionWrapperState(
            cell_state=cell_state,
            time=tf.zeros([], dtype=tf.int32),
            attention=tf.zeros(shape=[batch_size, self.attention_layer_size], dtype = dtype),
            alignments= self.attention_mechanism.initial_alignments(batch_size, dtype),
            alignment_history=())


    def call(self, inputs, state):
        cell_inputs = self.cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self.cell(cell_inputs, cell_state)
        cell_output = tf.identity(cell_output, name="checked_cell_output")

        previous_alignment = state.alignments
        previous_alignment_history = state.alignment_history

        attention_mechanism = self.attention_mechanism
        attention, alignments = _compute_attention(attention_mechanism, cell_output, previous_alignment, self.attention_layer)
        alignment_history = previous_alignment_history.write(state.time, alignments) if self.use_alignment_history else ()

        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            alignments=alignments,
            alignment_history=(alignment_history))

        return attention, next_state

