BatchSize = 100
VocabSize = 10000
EmbeddingSize = 256
HiddenSize = 512
MaxLength = 60
BucketGap = 20
LearningRate = 0.001
ValiditionPerBatch = 10000

srcVocabF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/train/c.dict.txt"
trgVocabF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/train/e.dict.txt"
trainSrcF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/train/c.txt"
trainTrgF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/train/e.txt"
valSrcF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/valid/c.txt"
valTrgF = "D:/Users/Shujliu/TensorflowNMT/IWSLT/valid/e.txt"


#trgVocabF = "D:/Users/Shujliu/TensorflowLM/PennLM/ptb.vocab"
#trainTrgF = "D:/Users/Shujliu/TensorflowLM/PennLM/ptb.train.txt"
#valTrgF = "D:/Users/Shujliu/TensorflowLM/PennLM/ptb.valid.txt"
#modelF = "D:/Users/Shujliu/TensorflowLM/PennLM/tensorflow.model"
