from __future__ import print_function
import os
import tensorflow as tf
import math
import Config
import NMT_Model
import Corpus
import nltk
import numpy

class NMT_Trainer:

    def __init__(self):
        self.model = NMT_Model.NMT_Model()
        self.srcVocab = Corpus.Vocabulary()
        self.trgVocab = Corpus.Vocabulary()
        self.srcVocab.loadDict(Config.srcVocabF)
        self.trgVocab.loadDict(Config.trgVocabF)
        self.trainData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.trainSrcF, Config.trainTrgF)
        self.valData = Corpus.BiCorpus(self.srcVocab, self.trgVocab, Config.valSrcF, Config.valTrgF)
        self.bestValCE = 999999
        self.bestBleu = 0
        self.badValCount = 0
        self.maxBadVal = 5
        self.learningRate = Config.LearningRate

        self.inputSrc = tf.placeholder(tf.int32, shape=[None, Config.BatchSize],
                                       name='srcInput')
        self.inputSrcR = tf.placeholder(tf.int32, shape=[None, Config.BatchSize],
                                       name='srcInputR')
        self.lengthSrc = tf.placeholder(tf.int32, shape=Config.BatchSize, name='lengthSrc')
        self.maskSrc = tf.placeholder(tf.float32, shape=[None, Config.BatchSize], name='srcMask')


        self.inputTrg = tf.placeholder(tf.int32, shape=[None, Config.BatchSize],
                                       name='trgInput')
        self.outputTrg = tf.placeholder(tf.int32, shape=[None, Config.BatchSize],
                                       name='trgOutput')
        self.maskTrg = tf.placeholder(tf.float32, shape=[None, Config.BatchSize], name='trgMask')
        self.lengthTrg = tf.placeholder(tf.int32, shape=Config.BatchSize, name='lengthTrg')


        self.optimizer = tf.train.AdamOptimizer()


    def train(self):
        min_loss, loss = self.model.network(self.inputSrc, self.outputTrg, self.maskTrg, self.lengthSrc,
                                            self.lengthTrg, self.optimizer)
        cePerWordBest = 10000
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        init = tf.local_variables_initializer()
        sess.run(init)

        for i in range(0, 100000000, 1):
            print("Training with batch " + str(i), end="\r")

            trainBatch = self.trainData.getTrainBatch()
            (batchSrc, batchTrg, lengthSrc, lengthTrg, srcMask, trgMask) \
                = self.trainData.buildInput(trainBatch)
            train_dict = {self.inputSrc:batchSrc, self.outputTrg: batchTrg,
                          self.lengthSrc: lengthSrc, self.lengthTrg: lengthTrg, self.maskSrc: srcMask, self.maskTrg: trgMask}
            _, cePerWord = sess.run([min_loss, loss], feed_dict=train_dict)
            if (i % 10 == 0):
                print(str(cePerWord / math.log(2.0)))


if __name__ == '__main__':
    nmtTrainer = NMT_Trainer()
    nmtTrainer.train()


