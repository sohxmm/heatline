"""Independent derivatives, numerical fixture, optimizer and reproducibility checks."""
import json
import numpy as np
import pytest
from src.ann_numpy import EBPANetwork
from src.lab_examples import backpropagation_trace


@pytest.mark.parametrize('activation', ['binary_sigmoid', 'bipolar_sigmoid'])
@pytest.mark.parametrize('loss', ['half_squared_error', 'binary_cross_entropy'])
@pytest.mark.parametrize('pos_weight', [1.0, 3.5])
def test_finite_difference_all_weights_and_biases(activation, loss, pos_weight):
    net = EBPANetwork(3, 2, activation=activation, loss=loss, seed=7)
    X = np.array([[0.2, -0.3, 0.7], [-1.2, 0.5, 0.4], [0.1, 0.2, -0.5]])
    y = np.array([0, 1, 1])
    actual = net.gradients(X, y, pos_weight)
    epsilon = 1e-6
    for name in ('V', 'v0', 'W', 'w0'):
        value = getattr(net, name)
        numeric = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            initial = value[index]
            value[index] = initial + epsilon
            plus = net.loss_on_batch(X, y, pos_weight)
            value[index] = initial - epsilon
            minus = net.loss_on_batch(X, y, pos_weight)
            value[index] = initial
            numeric[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(actual[f'grad_{name}'], numeric, atol=2e-9, rtol=2e-6)


@pytest.mark.parametrize('activation', ['binary_sigmoid', 'bipolar_sigmoid'])
@pytest.mark.parametrize('loss', ['half_squared_error', 'binary_cross_entropy'])
def test_independent_pytorch_reference(activation, loss):
    torch = pytest.importorskip('torch')
    net = EBPANetwork(3, 2, activation=activation, loss=loss, seed=31)
    X = np.array([[0.3, 0.2, -0.1], [0.1, -0.8, 0.4]])
    y = np.array([0, 1])
    weight = 2.7
    gradients = net.gradients(X, y, weight)
    parameters = {name: torch.tensor(getattr(net, name), dtype=torch.float64, requires_grad=True)
                  for name in ('V', 'v0', 'W', 'w0')}
    inputs = torch.tensor(X, dtype=torch.float64)
    targets = torch.tensor(y[:, None], dtype=torch.float64)
    hidden_net = inputs @ parameters['V'] + parameters['v0']
    hidden = torch.sigmoid(hidden_net) if activation == 'binary_sigmoid' else torch.tanh(hidden_net / 2)
    logits = hidden @ parameters['W'] + parameters['w0']
    prediction = torch.sigmoid(logits)
    if loss == 'half_squared_error':
        objective = (0.5 * (1 + (weight - 1) * targets) * (prediction - targets) ** 2).mean()
    else:
        objective = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(weight, dtype=torch.float64))
    objective.backward()
    np.testing.assert_allclose(net.forward(X), prediction.detach().numpy(), atol=1e-14)
    assert net.loss_on_batch(X, y, weight) == pytest.approx(objective.item(), abs=1e-14)
    for name, value in parameters.items():
        np.testing.assert_allclose(gradients[f'grad_{name}'], value.grad.numpy(), atol=1e-14)


def test_independent_scalar_exp7_fixture():
    trace = backpropagation_trace()
    net = EBPANetwork(2, 2, activation='binary_sigmoid', learning_rate=0.25, momentum=0,
                      loss='half_squared_error')
    for name, values in trace['initial'].items():
        setattr(net, name, np.asarray(values).reshape(getattr(net, name).shape))
    prediction = net.forward(trace['inputs'])
    # Independently evaluated with 50-digit decimal exponentials.
    assert prediction.item() == pytest.approx(0.5810319481280128834, abs=1e-15)
    result = net.backward(trace['inputs'], [1], prediction, clip_value=None)
    assert result['delta_k'].item() == pytest.approx(0.1019909947423485210, abs=1e-15)
    assert result['delta_k'].item() == pytest.approx(trace['backward']['delta_output'], abs=1e-15)
    for name, values in trace['updated'].items():
        np.testing.assert_allclose(getattr(net, name), np.asarray(values).reshape(getattr(net, name).shape), atol=1e-15)
    assert net.loss_on_batch(trace['inputs'], [1]) == pytest.approx(trace['loss_after'])
    assert trace['loss_after'] < trace['loss_before']
    assert trace['mu'] == 0 and trace['eta'] == 0.25


def test_momentum_and_positive_weight_use_same_objective():
    net = EBPANetwork(2, 2, loss='binary_cross_entropy', learning_rate=0.2, momentum=0.7)
    X, y = np.array([[0.1, 0.2], [-0.3, 0.1]]), [1, 0]
    previous = {}
    for step in range(2):
        grads = net.gradients(X, y, pos_weight=4)
        old = {name: getattr(net, name).copy() for name in ('V', 'v0', 'W', 'w0')}
        net.backward(X, y, pos_weight=4, clip_value=None)
        for name in old:
            expected = -0.2 * grads[f'grad_{name}'] + 0.7 * previous.get(name, 0)
            np.testing.assert_allclose(getattr(net, name) - old[name], expected, atol=1e-16)
            previous[name] = expected


def test_bce_extreme_logits_are_finite_and_not_clipped_gradients():
    net = EBPANetwork(1, 1, loss='bce')
    net.W[:] = 0
    net.w0[:] = 1000
    assert net.loss_on_batch([[0]], [0]) == pytest.approx(1000)
    assert net.gradients([[0]], [0])['grad_w0'].item() == pytest.approx(1)


@pytest.mark.parametrize('suffix', ['json', 'npz'])
def test_serialization_reinitializes_shapes_and_records_objective(tmp_path, suffix):
    net = EBPANetwork(3, 8, activation='binary_sigmoid', loss='bce', seed=13)
    X = np.array([[0.1, 0.2, 0.3]])
    path = tmp_path / f'model.{suffix}'
    net.save_weights(path)
    loaded = EBPANetwork(1, 1)
    loaded.load_weights(path)
    np.testing.assert_array_equal(net.predict_proba(X), loaded.predict_proba(X))
    assert loaded.loss == 'binary_cross_entropy'
    assert loaded.delta_V_prev.shape == net.V.shape
    assert not loaded.legacy_inconsistent_loss_reporting
    loaded.backward(X, [1])  # shape-changing loading supports further updates


def test_legacy_weights_load_with_explicit_reporting_flag(tmp_path):
    net = EBPANetwork(2, 3)
    path = tmp_path / 'legacy.json'
    net.save_weights(path)
    data = json.loads(path.read_text())
    del data['loss']
    del data['legacy_inconsistent_loss_reporting']
    path.write_text(json.dumps(data))
    net.load_weights(path)
    assert net.loss == 'half_squared_error'
    assert net.legacy_inconsistent_loss_reporting


def test_xor_learning_and_seeded_determinism():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    nets = [EBPANetwork(2, 8, activation='bipolar_sigmoid', learning_rate=0.5,
                        momentum=0.8, seed=7, loss='half_squared_error') for _ in range(2)]
    histories = [net.train(X, y, max_epochs=2000, batch_size=4, pos_weight=1) for net in nets]
    assert np.array_equal(nets[0].predict(X), y)
    assert histories[0] == histories[1]
    np.testing.assert_array_equal(nets[0].V, nets[1].V)
    assert nets[0].loss_on_batch(X, y) < 0.005
