# z

import numpy as np

# Sigmoid and tanh functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Given parameters
W_f_h = 0
W_f_x = 0
b_f = -100

W_i_h = 0
W_i_x = 100
b_i = 100

W_o_h = 0
W_o_x = 100
b_o = 0

W_c_h = -100
W_c_x = 50
b_c = 0

# Initial conditions
h_prev = 0
c_prev = 0

# Input sequence
x = [1, 1, 0, 1, 1]


# Initialize arrays to store the results
h = []
c = []

# Compute h_t and c_t for each time step
for t in range(len(x)):
    f_t = sigmoid((W_f_h * h_prev)+ (W_f_x * x[t]) + b_f)
    i_t = sigmoid((W_i_h * h_prev) + (W_i_x * x[t]) + b_i)
    o_t = sigmoid((W_o_h * h_prev) + (W_o_x * x[t]) + b_o)
    c_t = (f_t * c_prev) + (i_t * tanh((W_c_h * h_prev) + (W_c_x * x[t]) + b_c))
    h_t = o_t * tanh(c_t)
    
    # Append results to the lists
    h.append(h_t)
    c.append(c_t)
    
    # Update previous states
    h_prev = h_t
    c_prev = c_t

# Print the results
print("h_t values:", h)
# print(tanh(1))
# print(sigmoid(-1.15))