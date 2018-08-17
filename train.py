import tensorflow as tf
import math

# initializing data, just a bunch of sequences of numbers made up of patterns.
# this is what we want our RNN to learn to predict
data = [
    [1, 2, 3, 4, 5, 6],
    [2, 4, 6, 8],
    [5, 8, 11, 14, 17],
    [7, 8, 9, 10, 11, 12],
    [2, 4, 8, 16, 32, 64, 128],
    [1, 1, 2, 3, 5, 8, 13, 21, 34],
    [1, 1, 1, 1, 1, 1, 1]
]

# the length of the longest sequence we have - used to pad our sequences
# in tensorflow you have to pad variable length sequences during RNN training, because it rolls out the RNN
MAX_SEQ_LEN = max(map(len, data))

# these will be the variables that contain the padded data that are also of the shape:
# [batch, timestep, data]
X = []
Y = []

# we keep track of the original length of the sequences as well (before the padding). tensorflow can use this to
# be more efficient, and so the model doesn't actually use the padded 0s
SEQ_LENS = []

# pad the data
for i, sequence in enumerate(data):
    padded_sequence = sequence + [0] * (MAX_SEQ_LEN - len(sequence))

    # the target values will just be the next element in the sequence, therefore the Ys are just offset by 1
    X.append([[v] for v in padded_sequence[:-1]])
    Y.append([[v] for v in padded_sequence[1:]])

    # because we are removing an element when we add to X and Y, the sequence length is actually 1 smaller
    SEQ_LENS.append(len(sequence) - 1)


# our LSTM class - mostly convenience, but also to mirror the realtime.py setup
class TrainLSTM(object):
    def __init__(self, x, units):
        """
        :param x: the input tensor
        :param units: the size of the LSTM
        """
        # seq_length will be what we stuff SEQ_LENS in to tell tensorflow the actual length of the padded sequence
        self.seq_length = tf.placeholder(tf.int32, shape=[None])

        # standard dynamic_rnn initialization. note how we pass self.seq_length into sequence_length
        self.output, lstm_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicLSTMCell(units), x, dtype=tf.float32,
                                                    sequence_length=self.seq_length)


# convenience class - again mostly conveience and to mirror realtime.py setup
class TrainModel(object):
    def __init__(self, scope):
        # scope the variables so we can save & load them properly
        with tf.variable_scope(scope):
            # input will have the shape [batch, timestep, data]. since our data is just 1 integer at a time, we put
            # shape 1 for our data
            self.input = tf.placeholder(tf.float32, [None, None, 1])

            # use our lstm conveience class
            self.lstm = TrainLSTM(self.input, 16)

            # we are only outputting 1 number - the next number in the sequence
            self.output = tf.layers.dense(self.lstm.output, 1)

        # these are our training tensors - target is where we will put our target predictions Y, and loss is how
        # we compute the loss (just using mean squared error right now)
        self.target = tf.placeholder(tf.float32, [None, None, 1])
        self.loss = tf.reduce_mean(tf.square(self.output - self.target))


# main method
def train():
    # initialize the model with the scope 'seq-model'. all variables will be under that scope
    model = TrainModel('seq-model')

    # create an optimizer and capture the train_step we will use to train the model
    train_step = tf.train.AdamOptimizer().minimize(model.loss)

    # create a saver so we can save off the weights of the network and use it in realtime.py
    # note: not sure if passing in variables is necessary
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'seq-model')
    saver = tf.train.Saver(variables)

    with tf.Session() as session:
        # initialize all the weights
        session.run(tf.global_variables_initializer())

        # our main training loop - keeps going until our loss (mean squared error) gets below 1
        i = 0
        loss = math.inf
        while loss > 1:
            # compute the loss and run the training step.
            # two things:
            # 1. no batching is done here, we just pass in all of our data. this is because we have such a small data
            #    set and this is a toy example
            # 2. note how we have to pass in SEQ_LENS to the seq_length place holder
            loss, _ = session.run([model.loss, train_step],
                                  {model.input: X, model.target: Y, model.lstm.seq_length: SEQ_LENS})
            i += 1
            print(i, loss)

        # save all the weights
        saver.save(session, 'model/model.ckpt')

        # do some testing to show how we do - pass in part of the sequence and make sure the number it predicts is the
        # next number in the sequence
        for sequence in X:
            for i in range(len(sequence) - 1):
                if i == 0: continue
                print(sequence[:i])
                print(session.run([model.output], {model.input: [sequence[:i]], model.lstm.seq_length: [i]})[0][0][-1])


if __name__ == '__main__':
    train()
