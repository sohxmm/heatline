"""Executable academic appendix: the supplied seven lab experiments and COs.

These are guided examples, not seven product modules. All numbers in the
Experiment 7 trace are derived by scalar arithmetic independently of the ANN.
The supplied worksheet does not specify initial weights here; this module's
weights are an explicitly labelled illustrative fixture, not worksheet claims.
"""
from __future__ import annotations
import math
import numpy as np

LAB_MAPPING = [
    {'experiment': 1, 'title': 'Activation functions', 'co': 'CO1', 'connection': 'Identity, binary/bipolar step and sigmoid, ramp; compare sigmoid hidden activations.'},
    {'experiment': 2, 'title': 'McCulloch–Pitts neuron', 'co': 'CO1', 'connection': 'Fixed-threshold NAND, NOR and recurrent memory; no learning.'},
    {'experiment': 3, 'title': 'Hebbian learning', 'co': 'CO3', 'connection': 'Binary/bipolar association through weight updates.'},
    {'experiment': 4, 'title': 'Perceptron learning', 'co': 'CO2', 'connection': 'AND is separable; XOR is not. A single-layer weather comparator.'},
    {'experiment': 5, 'title': 'Delta learning', 'co': 'CO2', 'connection': 'Error-driven sigmoid updates; distinguish from linear LMS/ADALINE.'},
    {'experiment': 6, 'title': 'Multilayer perceptron', 'co': 'CO2', 'connection': 'A hidden layer learns XOR; the heatwave model uses the same architecture principle.'},
    {'experiment': 7, 'title': 'Error backpropagation', 'co': 'CO2', 'connection': 'Numerical 2–2–1 trace; α=0.25 denotes η; μ=0.'},
]


def activation_values(x):
    """Experiment 1 / CO1, including explicit step-at-zero and ramp conventions."""
    x = np.asarray(x, dtype=float)
    sigmoid = np.exp(-np.logaddexp(0, -x))
    return {'identity': x, 'binary_step': (x >= 0).astype(float),
            'bipolar_step': np.where(x >= 0, 1.0, -1.0),
            'binary_sigmoid': sigmoid, 'bipolar_sigmoid': np.tanh(x / 2),
            'ramp': np.clip(x, 0, 1)}


def mcp_examples():
    """Experiment 2: y=1 iff weighted sum+bias >= 0; reset-dominant memory."""
    logic = []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        logic.append({'inputs': [a, b], 'nand': int(-a - b + 1.5 >= 0),
                      'nor': int(-a - b + 0.5 >= 0)})
    q = 0
    memory = []
    for set_input, reset in [(0, 0), (1, 0), (0, 0), (0, 0), (0, 1), (0, 0)]:
        previous = q
        q = int(set_input + q - 2 * reset - 0.5 >= 0)
        memory.append({'set': set_input, 'reset': reset, 'previous': previous, 'next': q})
    return {'co': 'CO1', 'experiment': 2, 'logic': logic,
            'nand_weights': [-1, -1], 'nand_bias': 1.5,
            'nor_weights': [-1, -1], 'nor_bias': 0.5,
            'memory_rule': 'q_next = step(set + q_previous - 2*reset - 0.5)',
            'memory': memory}


def hebbian_example(bipolar=False, eta=1.0):
    """Experiment 3 / CO3: supervised Hebbian Δw=ηxt, Δb=ηt for AND.

    This illustrates association, not an error-minimizing classifier. Binary
    zeros produce no update; bipolar negatives contribute signed associations.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    target = np.array([0, 0, 0, 1], dtype=float)
    if bipolar:
        X, target = 2 * X - 1, 2 * target - 1
    weights, bias, steps = np.zeros(2), 0.0, []
    for x, t in zip(X, target):
        dw, db = eta * x * t, eta * t
        weights, bias = weights + dw, bias + db
        steps.append({'x': x.tolist(), 'target': float(t), 'delta_w': dw.tolist(),
                      'delta_b': float(db), 'weights': weights.tolist(), 'bias': float(bias)})
    return {'experiment': 3, 'co': 'CO3', 'encoding': 'bipolar' if bipolar else 'binary',
            'eta': eta, 'steps': steps, 'weights': weights.tolist(), 'bias': float(bias)}


def delta_example(bipolar=False):
    """One explicitly calculated sigmoid-delta update, Experiment 5 / CO2."""
    x, weights, bias, eta, target = [1.0, -1.0], [0.2, -0.1], 0.1, 0.25, 1.0
    net = sum(a * b for a, b in zip(x, weights)) + bias
    output = math.tanh(net / 2) if bipolar else 1 / (1 + math.exp(-net))
    derivative = 0.5 * (1 + output) * (1 - output) if bipolar else output * (1 - output)
    delta = (target - output) * derivative
    return {'experiment': 5, 'co': 'CO2', 'encoding': 'bipolar' if bipolar else 'binary',
            'inputs': x, 'initial_weights': weights, 'initial_bias': bias, 'eta': eta,
            'target': target, 'net': net, 'output': output, 'derivative': derivative,
            'delta': delta, 'updated_weights': [w + eta * delta * a for w, a in zip(weights, x)],
            'updated_bias': bias + eta * delta,
            'linear_lms_distinction': 'ADALINE uses output=net and derivative=1. This sigmoid loss is not quadratic in weights.'}


def backpropagation_trace():
    """Independent scalar 2–2–1 fixture for Experiment 7 / CO2.

    V rows correspond to inputs and columns to hidden neurons. All hidden error
    signals use the original W, before any parameter updates. The loss is half
    squared error, η=α=0.25 and μ=0. Biases are updated along with every weight.
    """
    x, target, eta = [0.0, 1.0], 1.0, 0.25
    V, v0, W, w0 = [[0.1, 0.2], [0.3, 0.4]], [0.1, -0.1], [0.5, -0.3], 0.2
    sigmoid = lambda n: 1 / (1 + math.exp(-n))
    hidden_net = [sum(x[i] * V[i][j] for i in range(2)) + v0[j] for j in range(2)]
    hidden = [sigmoid(n) for n in hidden_net]
    output_net = sum(hidden[j] * W[j] for j in range(2)) + w0
    output = sigmoid(output_net)
    error = target - output
    delta_output = error * output * (1 - output)
    delta_hidden = [delta_output * W[j] * hidden[j] * (1 - hidden[j]) for j in range(2)]
    dW = [eta * delta_output * z for z in hidden]
    dw0 = eta * delta_output
    dV = [[eta * delta_hidden[j] * x[i] for j in range(2)] for i in range(2)]
    dv0 = [eta * d for d in delta_hidden]
    updated_V = [[V[i][j] + dV[i][j] for j in range(2)] for i in range(2)]
    updated_v0 = [v0[j] + dv0[j] for j in range(2)]
    updated_W, updated_w0 = [W[j] + dW[j] for j in range(2)], w0 + dw0
    new_hidden = [sigmoid(sum(x[i] * updated_V[i][j] for i in range(2)) + updated_v0[j]) for j in range(2)]
    new_output = sigmoid(sum(new_hidden[j] * updated_W[j] for j in range(2)) + updated_w0)
    return {'experiment': 7, 'co': 'CO2', 'fixture': 'illustrative-exp7-v1',
            'initial_weights_note': 'Explicit illustrative weights; not supplied worksheet weights.',
            'inputs': x, 'target': target, 'eta': eta, 'worksheet_alpha': eta, 'mu': 0.0,
            'loss': 'half_squared_error', 'activation': 'binary_sigmoid',
            'initial': {'V': V, 'v0': v0, 'W': W, 'w0': w0},
            'forward': {'hidden_net': hidden_net, 'hidden': hidden, 'output_net': output_net, 'output': output},
            'backward': {'error': error, 'delta_output': delta_output, 'delta_hidden': delta_hidden},
            'updates': {'V': dV, 'v0': dv0, 'W': dW, 'w0': dw0},
            'updated': {'V': updated_V, 'v0': updated_v0, 'W': updated_W, 'w0': updated_w0},
            'loss_before': 0.5 * error ** 2, 'loss_after': 0.5 * (target - new_output) ** 2,
            'output_after': new_output}


def appendix_examples():
    """JSON-ready content for the notebook, report and learning walkthrough."""
    grid = np.linspace(-6, 6, 121)
    return {'syllabus': LAB_MAPPING,
            'activations': {'x': grid.tolist(), **{k: v.tolist() for k, v in activation_values(grid).items()},
                            'conventions': 'Step includes zero in positive class; ramp=clip(x,0,1).'},
            'mcp': mcp_examples(), 'hebbian': [hebbian_example(False), hebbian_example(True)],
            'delta': [delta_example(False), delta_example(True)],
            'backpropagation': backpropagation_trace()}
