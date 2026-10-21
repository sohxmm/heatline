import numpy as np
from src.lab_examples import LAB_MAPPING, activation_values, mcp_examples, hebbian_example, delta_example
from src.sc_baselines import PerceptronRuleClassifier, SigmoidDeltaRuleClassifier, DeltaRuleADALINE


def test_lab_printed_co_mapping():
    assert [row['experiment'] for row in LAB_MAPPING] == list(range(1, 8))
    assert [row['co'] for row in LAB_MAPPING] == ['CO1', 'CO1', 'CO3', 'CO2', 'CO2', 'CO2', 'CO2']


def test_all_supplied_activations_and_conventions():
    functions = activation_values([-2, 0, 0.5, 2])
    assert set(functions) == {'identity', 'binary_step', 'bipolar_step', 'binary_sigmoid', 'bipolar_sigmoid', 'ramp'}
    np.testing.assert_array_equal(functions['ramp'], [0, 0, 0.5, 1])
    np.testing.assert_array_equal(functions['binary_step'], [0, 1, 1, 1])
    np.testing.assert_allclose(functions['bipolar_sigmoid'], 2 * functions['binary_sigmoid'] - 1)


def test_fixed_mcp_logic_and_recurrent_memory():
    examples = mcp_examples()
    assert [row['nand'] for row in examples['logic']] == [1, 1, 1, 0]
    assert [row['nor'] for row in examples['logic']] == [1, 0, 0, 0]
    assert [row['next'] for row in examples['memory']] == [0, 1, 1, 1, 0, 0]


def test_hebbian_binary_bipolar_worked_updates():
    binary, bipolar = hebbian_example(), hebbian_example(True)
    assert binary['weights'] == [1, 1] and binary['bias'] == 1
    assert bipolar['weights'] == [2, 2] and bipolar['bias'] == -2


def test_and_converges_but_single_layer_cannot_fit_xor():
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    and_model = PerceptronRuleClassifier(max_epochs=50).fit(X, [0, 0, 0, 1])
    np.testing.assert_array_equal(and_model.predict(X), [0, 0, 0, 1])
    xor_model = PerceptronRuleClassifier(max_epochs=50).fit(X, [0, 1, 1, 0])
    assert np.any(xor_model.predict(X) != [0, 1, 1, 0])


def test_sigmoid_delta_rule_matches_independent_single_update():
    # At zero initialization and x=[1,-1], binary y=.5, delta=.125.
    model = SigmoidDeltaRuleClassifier(eta=.25, max_epochs=1).fit([[1, -1]], [1])
    np.testing.assert_allclose(model.w_, [.03125, -.03125])
    assert model.b_ == .03125
    bipolar = SigmoidDeltaRuleClassifier(eta=.25, max_epochs=1, activation='bipolar_sigmoid').fit([[1, -1]], [1])
    np.testing.assert_allclose(bipolar.w_, [.125, -.125])
    assert bipolar.b_ == .125
    assert DeltaRuleADALINE is SigmoidDeltaRuleClassifier
    assert delta_example()['updated_bias'] > delta_example()['initial_bias']
