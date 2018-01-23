import tensorflow as tf
import ZoneOutLSTM
import Config
import AttentionWrapper

class NMT_Model:

    def __init__(self):
        self.scope = "NMT_Model"
        with tf.variable_scope(self.scope):
            self.EmbSrc = tf.get_variable("SrcEmbedding", shape=[Config.VocabSize, Config.EmbeddingSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.EmbTrg = tf.get_variable("TrgEmbedding", shape=[Config.VocabSize, Config.EmbeddingSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.EncoderL2R = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.HiddenSize,
                                                          scope=self.scope+"_EncoderL2R")
            self.EncoderR2L = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize, hidden_size=Config.HiddenSize,
                                                          scope=self.scope+"_EncoderR2L")
            self.Decoder = ZoneOutLSTM.ZoneoutLSTMCell(input_size=Config.EmbeddingSize+2*Config.HiddenSize,
                                                       hidden_size=Config.HiddenSize, scope=self.scope+"_Decoder")

            self.Wt = tf.get_variable("ReadoutWeight", shape=[Config.HiddenSize + Config.EmbeddingSize, Config.VocabSize],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.Wtb = tf.get_variable("ReadoutBias", shape=Config.VocabSize, initializer=tf.constant_initializer(0.0))

            self.WI = tf.get_variable("DecoderInitWeight",shape=(Config.HiddenSize, Config.HiddenSize*2),
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.WIb = tf.get_variable("DecoderInitBias", shape=Config.HiddenSize*2, initializer=tf.constant_initializer(0.0))


        self.Parameters = [self.EmbSrc, self.EmbTrg, self.Wt, self.Wtb, self.WI, self.WIb]
        self.Parameters.extend(self.EncoderL2R.Parameters)
        self.Parameters.extend(self.EncoderR2L.Parameters)
        self.Parameters.extend(self.Decoder.Parameters)


    def buildEncoderStates(self, inputSrc, srcLength):
        inputSrcEmbed = tf.nn.embedding_lookup(self.EmbSrc, inputSrc)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(self.EncoderL2R, self.EncoderL2R, inputSrcEmbed, dtype=tf.float32,
                                                               sequence_length=srcLength, time_major=True)
        outputs = tf.concat(bi_outputs, -1)
        return outputs, bi_state[1][0]

    def buildDecoderInitState(self, srcSentEmb):
        WIS = tf.matmul(srcSentEmb, self.WI) + self.WIb
        initHiddenMem = tf.tanh(WIS)
        initHiddden, initMem = tf.split(initHiddenMem, 2, -1)
        return tf.nn.rnn_cell.LSTMStateTuple(initMem, initHiddden)

    def buildAttentionCell(self, encoderOutputs, sourceSequenceLength):
        memory = tf.transpose(encoderOutputs, [1, 0, 2])

        attention_mechanism = AttentionWrapper.MyAttention(num_units=Config.HiddenSize, memory=memory,
                                                                   memory_sequence_length=sourceSequenceLength)
        cell = ZoneOutLSTM.ZoneoutLSTMCell(Config.HiddenSize + Config.EmbeddingSize, Config.HiddenSize)
        attn_cell = AttentionWrapper.MyAttentionWrapper(cell, attention_mechanism, attention_layer_size=Config.HiddenSize)
        return attn_cell

    def createDecoderNetwork(self, srcrHiddens, srcSentEmb, outputTrg, maskTrg, srcLength, trgLength, optimizer):

        decoderInitState = self.buildDecoderInitState(srcSentEmb)
        decoderCell = self.buildAttentionCell(srcrHiddens, srcLength)
        decoderInitState = decoderCell.zero_state(Config.BatchSize, tf.float32).clone(cell_state=decoderInitState)

        inputTrgEmbed = tf.nn.embedding_lookup(self.EmbTrg, outputTrg)
        helper = tf.contrib.seq2seq.TrainingHelper(inputTrgEmbed, trgLength, time_major=True)
        my_decoder = tf.contrib.seq2seq.BasicDecoder(decoderCell, helper, decoderInitState)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major=True,
                                                                            swap_memory=True, scope=self.scope)
        outputs = outputs.rnn_output
        readOut = tf.concat([outputs, inputTrgEmbed], axis=-1)
        readOut = tf.reshape(readOut, shape=[-1, Config.HiddenSize + Config.EmbeddingSize])
        logits_out = tf.matmul(readOut, self.Wt) + self.Wtb
        outputTrg = tf.reshape(outputTrg, shape=[-1])
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=outputTrg)
        maskTrg = tf.reshape(maskTrg, shape=[-1])
        tce = tf.reduce_sum(ce * maskTrg)
        totalCount = tf.reduce_sum(maskTrg)
        tce = tce / totalCount
        min_loss = optimizer.minimize(tce)
        return min_loss, tce

    def createEncoderDecoderNetwork(self, inputSrc, outputTrg, maskTrg, srcLength, trgLength, optimizer):
        srcrHiddens, sentEmb = self.buildEncoderStates(inputSrc, srcLength)
        decoderNet = self.createDecoderNetwork(srcrHiddens, sentEmb, outputTrg, maskTrg, srcLength, trgLength, optimizer)
        return decoderNet
