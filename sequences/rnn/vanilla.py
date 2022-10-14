import numpy as np

""" RNN Pseudocode
state_t = 0
for input_t in input_sequence:
    output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
"""

timesteps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

"""
Each timestep t in the output tensor contains information about timesteps 0
to t in the input sequence—about the entire past. For this reason, in many
cases, you don’t need this full sequence of outputs; you just need the last out-
put ( output_t at the end of the loop), because it already contains informa-
tion about the entire sequence."""
final_output_sequence = successive_outputs[-1]

print()
