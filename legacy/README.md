# Original lab experiment

`original-experiment.zip` preserves the tracked source, configuration, processed
data, model weights, notebooks and reported outputs before the Heatline overhaul.
Raw hourly data remain in `data/raw/`. Extract the archive into a separate folder
and copy that raw directory if reproducing the old experiment.

These results used adjusted labels (43.5 °C or 40 °C with a 3.5 °C departure),
1991–2020 normals, and an ANN whose squared-error updates were reported as
weighted cross-entropy. They are legacy results, not evidence for the corrected
model. The deeper PyTorch ensemble, fuzzy inference and GA are optional theory
extensions; their old metrics must not be compared directly with the new target.
