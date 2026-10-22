# Academic appendix: lab experiments and printed COs

| experiment | title | co | connection |
| --- | --- | --- | --- |
| 1 | Activation functions | CO1 | Identity, binary/bipolar step and sigmoid, ramp; compare sigmoid hidden activations. |
| 2 | McCulloch–Pitts neuron | CO1 | Fixed-threshold NAND, NOR and recurrent memory; no learning. |
| 3 | Hebbian learning | CO3 | Binary/bipolar association through weight updates. |
| 4 | Perceptron learning | CO2 | AND is separable; XOR is not. A single-layer weather comparator. |
| 5 | Delta learning | CO2 | Error-driven sigmoid updates; distinguish from linear LMS/ADALINE. |
| 6 | Multilayer perceptron | CO2 | A hidden layer learns XOR; the heatwave model uses the same architecture principle. |
| 7 | Error backpropagation | CO2 | Numerical 2–2–1 trace; α=0.25 denotes η; μ=0. |

Use `notebooks/03_heatline_lab.ipynb` for executed activation, M-P, Hebbian, perceptron, delta and XOR examples. This appendix records the independently computed Experiment 7 fixture.

## Experiment 7 · CO2

Inputs [0,1], target 1, binary sigmoid; α=η=0.25, μ=0. Matrix V rows are inputs, columns hidden neurons. The initial numbers below are illustrative weights explicitly selected for this fixture. All hidden signals use original W.

```json
{
  "experiment": 7,
  "co": "CO2",
  "fixture": "illustrative-exp7-v1",
  "initial_weights_note": "Explicit illustrative weights; not supplied worksheet weights.",
  "inputs": [
    0.0,
    1.0
  ],
  "target": 1.0,
  "eta": 0.25,
  "worksheet_alpha": 0.25,
  "mu": 0.0,
  "loss": "half_squared_error",
  "activation": "binary_sigmoid",
  "initial": {
    "V": [
      [
        0.1,
        0.2
      ],
      [
        0.3,
        0.4
      ]
    ],
    "v0": [
      0.1,
      -0.1
    ],
    "W": [
      0.5,
      -0.3
    ],
    "w0": 0.2
  },
  "forward": {
    "hidden_net": [
      0.4,
      0.30000000000000004
    ],
    "hidden": [
      0.598687660112452,
      0.574442516811659
    ],
    "output_net": 0.3270110750127283,
    "output": 0.5810319481280128
  },
  "backward": {
    "error": 0.41896805187198716,
    "delta_output": 0.10199099474234853,
    "delta_hidden": [
      0.012252216227858518,
      -0.007479763914712277
    ]
  },
  "updates": {
    "V": [
      [
        0.0,
        -0.0
      ],
      [
        0.0030630540569646296,
        -0.0018699409786780692
      ]
    ],
    "v0": [
      0.0030630540569646296,
      -0.0018699409786780692
    ],
    "W": [
      0.015265187498709509,
      0.014646990927979844
    ],
    "w0": 0.025497748685587133
  },
  "updated": {
    "V": [
      [
        0.1,
        0.2
      ],
      [
        0.3030630540569646,
        0.398130059021322
      ]
    ],
    "v0": [
      0.10306305405696463,
      -0.10186994097867808
    ],
    "W": [
      0.5152651874987095,
      -0.28535300907202016
    ],
    "w0": 0.22549774868558714
  },
  "loss_before": 0.08776711424470406,
  "loss_after": 0.08334624810780353,
  "output_after": 0.5917200761541084
}
```

The binary/bipolar delta examples use the derivative of their declared activation. ADALINE/LMS uses a linear output. An output sigmoid is a bounded score, not evidence of calibration. The memory example is a reset-dominant recurrent M-P convention; its truth sequence is displayed in the notebook.
