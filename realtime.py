import tensorflow as tf
import numpy
from train import data


# LSTM class for keeping track of all of the LSTM pieces of data
# theres a big difference between how we will use this lstm and how we used the lstm in training:
# for training, we passed a whole sequence of time steps in at once
# for real time, we are only passing in 1 time step at a time
#
# therefore we need to keep track of the internal lstm state in between the time steps
#
# note: the intended use of this is for use in things like games or robotics... you are only getting 1 time step at a
# time
class RealTimeLSTM(object):
    def __init__(self, x, units):
        """
        :param x: input tensor
        :param units: size of the lstm
        """
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(units)

        # here we initialize the placeholders for the internal state.
        # we will need to put data into self.state_input when we call session.run()
        cell_state = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        hidden_state = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_input = [cell_state, hidden_state]

        # here is what we'll use initially to pass into self.state_input, and we'll use this when we want to reset
        # the LSTMs state
        self.state_init = [numpy.zeros((1, lstm_cell.state_size.c), numpy.float32),
                           numpy.zeros((1, lstm_cell.state_size.h), numpy.float32)]

        # wrap the two placeholders in a state tuple so we can pass it to the dynamic_rnn
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

        # same way of initializing the dynamic_rnn as before BUT with one big difference:
        # no need to pass in a sequence_length
        self.output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state)

        # this holds the new LSTM state after we invoke the rnn
        # lstm_state is a tuple of (cell_state, hidden_state), each with the shape:
        # (batch, data).
        self.state_out = (lstm_state[0][:1, :], lstm_state[1][:1, :])


# our model
class RealtimeModel(object):
    def __init__(self, scope):
        # again we have to scope the variables the same way so loading works
        # note: the setup of the model is exactly the same
        with tf.variable_scope(scope):
            self.input = tf.placeholder(tf.float32, [None, None, 1])
            self.lstm = RealTimeLSTM(self.input, 16)
            self.output = tf.layers.dense(self.lstm.output, 1)

        # note: we don't have any loss functions or placeholders here, because we aren't training with this model

        # we have to keep track of the LSTMs current state. this gets set when forward() is invoked later, and reset
        # when reset_states() is called
        self.current_state = self.lstm.state_init

    def reset_states(self):
        """
        Resets the LSTMs state. Use when you are about to start a new sequence
        :return:
        """
        self.current_state = self.lstm.state_init

    def forward(self, session, x):
        """
        Run the model
        :param session: tf.Session
        :param x: input data to pass into self.input. needs to be of the shape [batch, time step, data]. here we
        only pass in 1 batch and 1 time step at a time, so essentially we pass in [[data]].
        :return: output of the network
        """
        # since we include both output & self.lstm.state_out in our fetches, this function returns both the output
        # of the network and also the new state of the LSTM. we save it into self.current_state
        # note: in the feed dict, we pass self.current_state into self.lstm.state_input
        output, self.current_state = session.run([self.output, self.lstm.state_out],
                                                 {self.input: x,
                                                  self.lstm.state_input[0]: self.current_state[0],
                                                  self.lstm.state_input[1]: self.current_state[1]})
        return output


def main():
    # initialize the model
    model = RealtimeModel('seq-model')

    # initialize a saver to load all the weights we saved in train.py
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'seq-model')
    saver = tf.train.Saver(variables)

    with tf.Session() as session:
        # instead of initializing the weights with a `session.run(tf.global_variables_initializer())`, we load
        # the weights using the saver
        saver.restore(session, 'model/model.ckpt')

        # test the sequences
        for sequence in data:
            # reset the internal LSTM state - we are starting a new sequence
            model.reset_states()
            print(sequence)
            for i, x in enumerate(sequence[:-1]):
                # we expect it to predict the next number in the sequence
                y = sequence[i + 1]

                # here's where we invoke model.forward to get it's prediction
                # note: how we wrap the input number to be of the shape [batch, timestep, data]
                # note: how we unwrap the output of the model, which is in the shape [batch, timestep, data]
                predicted_y = model.forward(session, [[[x]]])[0][0][0]
                print('{} -> {} (predicted {:.3f})'.format(x, y, predicted_y))
            print('---')


if __name__ == '__main__':
    main()
