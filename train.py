import tensorflow as tf

# initializing data
data = [
    [1, 2, 3, 4, 5, 6],
    [2, 4, 6, 8],
    [5, 8, 11, 14, 17],
    [7, 8, 9, 10, 11, 12],
    [2, 4, 8, 16, 32, 64, 128],
    [1, 1, 2, 3, 5, 8, 13, 21, 34],
    [1, 1, 1, 1, 1, 1, 1]
]

MAX_SEQ_LEN = max(map(len, data))

X = []
Y = []
SEQ_LENS = []

# padding data
for i, sequence in enumerate(data):
    padded_sequence = sequence + [0] * (MAX_SEQ_LEN - len(sequence))
    X.append([[v] for v in padded_sequence[:-1]])
    Y.append([[v] for v in padded_sequence[1:]])
    SEQ_LENS.append(len(sequence) - 1)


class TrainLSTM(object):
    def __init__(self, x, units):
        self.seq_length = tf.placeholder(tf.int32, shape=[None])
        self.output, lstm_state = tf.nn.dynamic_rnn(tf.nn.rnn_cell.BasicLSTMCell(units), x, dtype=tf.float32,
                                                    sequence_length=self.seq_length)


class TrainModel(object):
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, None, 1])
            self.lstm = TrainLSTM(self.input, 16)
            self.output = tf.layers.dense(self.lstm.output, 1)

        self.target = tf.placeholder(tf.float32, [None, None, 1])
        self.loss = tf.reduce_mean(tf.square(self.output - self.target))


def train():
    model = TrainModel('seq-model')
    train_step = tf.train.AdamOptimizer().minimize(model.loss)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'seq-model')
    saver = tf.train.Saver(variables)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        loss = 2

        i = 0
        while loss > 1:
            loss, _ = session.run([model.loss, train_step],
                                  {model.input: X, model.target: Y, model.lstm.seq_length: SEQ_LENS})
            i += 1
            print(i, loss)

        saver.save(session, 'model/model.ckpt')

        for sequence in X:
            for i in range(len(sequence) - 1):
                if i == 0: continue
                print(sequence[:i])
                print(session.run([model.output], {model.input: [sequence[:i]], model.lstm.seq_length: [i]})[0][0][-1])


if __name__ == '__main__':
    train()
