"""Build and execute the current syllabus-aligned notebook without retraining weather models."""
from pathlib import Path
import sys
import nbformat as nb
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text): cells.append(nb.v4.new_markdown_cell(text))
    def code(text): cells.append(nb.v4.new_code_cell(text))
    md('# Heatline — from a neuron to a May heatwave ANN\n\nAssessed scope: **Nagpur, May, NumPy backpropagation**. The seven supporting experiments use the supplied lab-printed COs. Weather dashboard integration is a product extension; fuzzy/GA/deeper ensemble are optional. Run from this notebook folder or the repository root. This notebook reads measured artifacts rather than silently retraining models.')
    code("from pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd() if (Path.cwd()/'src').exists() else Path.cwd().parent\nsys.path.insert(0,str(ROOT))\nfrom src.lab_examples import *\nfrom src.ann_numpy import EBPANetwork\nfrom src.sc_baselines import PerceptronRuleClassifier\nARTIFACTS=ROOT/'artifacts/heatline'\npd.DataFrame(LAB_MAPPING)")
    md('## 1 · Activation functions · CO1\nAn artificial neuron computes a weighted sum plus bias. Identity and ramp define continuous mappings; binary and bipolar steps are threshold decisions. Binary sigmoid is σ(a); bipolar sigmoid is 2σ(a)−1=tanh(a/2), with derivative (1−f(a)²)/2. Step at zero belongs to the positive class here; ramp is clipped to [0,1].')
    code("x=np.linspace(-6,6,241)\nfig,axes=plt.subplots(2,3,figsize=(12,6))\nfor ax,(name,values) in zip(axes.flat,activation_values(x).items()):\n    ax.plot(x,values); ax.set_title(name); ax.grid(alpha=.2)\nfig.tight_layout(); plt.show()")
    md('## 2 · McCulloch–Pitts · CO1\nFixed weights implement NAND and NOR. The memory example feeds the previous output back into the threshold neuron. It is a reset-dominant memory convention, displayed explicitly; it does not learn weights.')
    code("examples=mcp_examples()\ndisplay(pd.DataFrame(examples['logic']))\nprint(examples['memory_rule'])\ndisplay(pd.DataFrame(examples['memory']))")
    md('## 3 · Hebbian association · CO3\nFor the supervised association example Δw=ηxt, Δb=ηt. Binary zeros do not contribute; bipolar negatives can create signed associations. Hebbian learning does not minimize the classification error used by backpropagation.')
    code("for bipolar in [False,True]:\n    result=hebbian_example(bipolar)\n    print(result['encoding']); display(pd.DataFrame(result['steps']))")
    md('## 4 · Perceptron and separability · CO2\nA single threshold boundary can represent bipolar AND. XOR requires two disjoint positive regions and cannot be separated by one straight line. This establishes an architecture limitation; it does not prove any particular weather dataset needs a nonlinear model.')
    code("binary=np.array([[0,0],[0,1],[1,0],[1,1]],float)\nX=2*binary-1\nand_target=np.array([0,0,0,1])\nxor_target=np.array([0,1,1,0])\nfor name,target in [('AND',and_target),('XOR',xor_target)]:\n    net=PerceptronRuleClassifier(eta=.1,max_epochs=100,random_state=42).fit(X,target)\n    print(name,'bipolar inputs:',X.tolist(),'bipolar targets:',(2*target-1).tolist(),'binary predictions:',net.predict(X).tolist())")
    md('## 5 · Delta learning · CO2\nFor sigmoid output and half-squared-error the update signal is (t−ŷ)f′(a). Binary targets are 0/1; bipolar sigmoid uses −1/+1 targets and its different derivative. True LMS/ADALINE uses a linear output a and derivative 1. Applying sigmoid and its derivative is a nonlinear sigmoid delta neuron, even if an old filename used ADALINE.')
    code("for bipolar in [False,True]:\n    print(json.dumps(delta_example(bipolar),indent=2))")
    md('## 6 · Multilayer XOR · CO2\nThe hidden layer permits a nonlinear decision boundary. The assessed climate network has 26 lagged inputs, one hidden layer (8,16,32 compared), and one binary sigmoid output. The XOR example below uses the same forward/backward implementation; it is a mathematical demonstration rather than a weather performance claim.')
    code("xor=EBPANetwork(2,8,activation='bipolar_sigmoid',learning_rate=.3,momentum=.9,seed=42,loss='binary_cross_entropy')\nxor.train(binary,xor_target,max_epochs=2000,batch_size=4)\nprint('XOR scores:',xor.predict_proba(binary))\nassert np.array_equal(xor.predict(binary),xor_target)")
    md('## 7 · One 2–2–1 backpropagation update · CO2\nInputs [0,1], target 1, binary sigmoid. The supplied α=0.25 means learning rate **η=0.25**; momentum **μ=0**. Initial weights below are an explicitly chosen illustrative fixture because worksheet weights were not supplied. All hidden error signals use the old output weights; all gradients are calculated before applying updates.\n\nδₖ=(t−ŷ)ŷ(1−ŷ), δⱼ=δₖwⱼzⱼ(1−zⱼ). Then Δwⱼ=ηδₖzⱼ, Δvᵢⱼ=ηδⱼxᵢ; bias updates use input 1.')
    code("trace=backpropagation_trace()\nprint(json.dumps(trace,indent=2))\nassert abs(trace['forward']['output']-0.5810319481280129)<1e-14\nassert trace['loss_after']<trace['loss_before']")
    md('## Consistent objectives and independent derivatives\nTextbook L=mean[½(ŷ−t)²] gives output-logit derivative (ŷ−t)ŷ(1−ŷ). BCE gives ŷ−t. Positive weight p multiplies the positive term in both loss and derivative: a(t)=1+(p−1)t, BCE derivative a(t)ŷ−pt. η controls step size; μ carries the previous parameter update. The primary study uses p=1.\n\nFinite differences and an independently written PyTorch float64 graph check every weight and bias for both sigmoid activations and both objectives. Run `python -m pytest tests/test_ann_gradients.py -q` for these checks; PyTorch is an independent reference, not the training implementation.')
    code("model=json.loads((ARTIFACTS/'model.json').read_text())\nexperiments=json.loads((ARTIFACTS/'experiments.json').read_text())\nprint('Architecture:',model['architecture'])\nprint('Selection:',model['selection_policy'])\ndisplay(pd.DataFrame(model['splits']))\ndisplay(pd.DataFrame(model['metrics']))")
    md('## Data and experiment interpretation\n1981–2010 trains the ANN, scaler and frozen reference normals; 2011–2014 selects the architecture/objective and early-stopping epoch; 2015–2018 fits Platt scaling and selects the F1 decision threshold. 2019–2026 is retrospective evaluation: these years were previously examined in the legacy project. April primes seven-day May lags. Missing dates and duplicate observations fail explicitly.\n\nThe target `nagpur-grid-proxy-v1` is Tmax≥45°C OR (Tmax≥40°C and reference departure≥4.5°C). It is an IMD-inspired single-grid daily proxy, not an official station-based spatial/persistence declaration. The old adjusted rule is preserved in the legacy archive, not chosen to force a desired prevalence.')
    code("runs=pd.DataFrame(experiments['runs'])\ndisplay(runs.groupby(['activation','hidden','loss'])['pr_auc'].agg(['mean','std','count']))\ndisplay(pd.DataFrame(experiments['missed_events'])[['date','score','probability','tmax','actual','prediction']])\ncurve=pd.DataFrame(model['training_history'])\ncurve.plot(x='epoch',y=['train_loss','val_loss'],figsize=(9,3));plt.ylabel(model['architecture']['loss']);plt.show()")
    md('## Results and limitations\nRead the measured table above, including cases where a simpler baseline wins. PR-AUC here means average precision. Brier score checks squared probability error; Platt scaling is an estimated transformation, not a guarantee of calibration. Only 8 positives occur in the 248 retrospective days, and neighboring days are correlated. Neither the model nor the provider weather dashboard is an official warning service. ERA5 is delayed; live/regional ANN output is unavailable until compatible timely inputs have separate validation.\n\nReports: `docs/ACADEMIC_REPORT.md`, `docs/ACADEMIC_APPENDIX.md`. Original source/results: `legacy/original-experiment.zip`.')
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    # Execute with the current interpreter via a temporary kernel specification.
    from tempfile import TemporaryDirectory
    from jupyter_client.kernelspec import KernelSpecManager
    with TemporaryDirectory() as temp:
        path=Path(temp)/'heatline';path.mkdir()
        (path/'kernel.json').write_text(__import__('json').dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Heatline','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[temp])
        client=NotebookClient(notebook,timeout=120,kernel_name='heatline',resources={'metadata':{'path':str(ROOT)}},kernel_manager_class=__import__('jupyter_client').KernelManager)
        client.km=__import__('jupyter_client').KernelManager(kernel_name='heatline',kernel_spec_manager=manager)
        client.execute()
    destination=ROOT/'notebooks/03_heatline_lab.ipynb'
    nb.write(notebook,destination)
    print(f'Executed {len(cells)} cells: {destination}')


if __name__=='__main__':
    build()
