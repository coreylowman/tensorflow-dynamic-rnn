# tensorflow-dynamic-rnn
Small toy example of a dynamic RNN in tensorflow, trained with variable length sequences, and run real time (single time step at a time)

# inspiration

I wanted to become more familiar with how to use RNNs for games. There are two messy things with games that make RNNs more complicated to use:

1. In games, you usually pass in the state each turn, and then expect to get some output. You can't wait for the game to end and then pass in the whole sequence, you have to invoke the RNN each time step
2. Not all games are the same length each time you play. For instance a game of chess can last 4 turns, or 50 turns.

Those are the problems I'm trying to investigate. So far I've found blog posts that talk about 1 or the other, but I've found very few resources
that discuss both.

# data

This repo just works with simple number sequences like

```
1,1,1,1,1
1,2,3,4
2,4,6,8,10
1,1,2,3,5,8
```

This is to make the data as simple as possibel to work with

# Problems

There are two parts to using a RNN: training it and running it.

## Training Problems

For training, we won't worry about issue #1, because we will have the whole sequence of time steps.
We do have to worry about issue #2 though, because of the way tensorflow trains RNNs.

## Running Problems

For running, we do have to worry about issue #1, beacuse again we don't have all the time steps.
We don't have to worry about issue #2, because we are only dealing with 1 game at a time.

# code

There are two files in this repo that address the above:

1. train.py, which does the following:
    1. initializes the model and the data we want to learn
    2. trains the model
    3. saves the model
2. realtime.py, which does the following:
    1. initialize the model
    2. loads the weights from the directory that the train.py script created
    3. runs the model on each sequence separately.
    

