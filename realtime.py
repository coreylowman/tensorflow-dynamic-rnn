import tensorflow as tf
import numpy
from train import data


class RealTimeLSTM(object):
    def __init__(self, x, units):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(units)

        cell_state = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        hidden_state = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

        self.output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state)
        self.state_input = [cell_state, hidden_state]
        self.state_init = [numpy.zeros((1, lstm_cell.state_size.c), numpy.float32),
                           numpy.zeros((1, lstm_cell.state_size.h), numpy.float32)]
        self.state_out = (lstm_state[0][:1, :], lstm_state[1][:1, :])


class RealtimeModel(object):
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, None, 1])
            self.lstm = RealTimeLSTM(self.input, 16)
            self.output = tf.layers.dense(self.lstm.output, 1)

        self.current_state = self.lstm.state_init

    def reset_states(self):
        self.current_state = self.lstm.state_init

    def forward(self, session, x):
        output, self.current_state = session.run([self.output, self.lstm.state_out],
                                                 {self.input: x,
                                                  self.lstm.state_input[0]: self.current_state[0],
                                                  self.lstm.state_input[1]: self.current_state[1]})
        return output


def main():
    model = RealtimeModel('seq-model')

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'seq-model')
    saver = tf.train.Saver(variables)

    with tf.Session() as session:
        saver.restore(session, 'model/model.ckpt')

        for sequence in data:
            model.reset_states()
            print(sequence)
            for i, x in enumerate(sequence[:-1]):
                print('{} -> {} (predicted {:.3f})'.format(x, sequence[i + 1], model.forward(session, [[[x]]])[0][0][0]))
            print('---')


if __name__ == '__main__':
    main()
