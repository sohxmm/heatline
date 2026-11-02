# Original lab experiment

`heatline-v1-experiment.zip` preserves the first mathematically corrected
May-only experiment before the seasonal v2 revision. It uses the same ERA5 May
targets as v2 and supplies the fair before/after comparison. Its primary model
detected 2/8 retrospective positives with two false alarms.

`original-experiment.zip` preserves the tracked source, configuration, processed
data, model weights, notebooks and reported outputs before the Heatline overhaul.
Raw hourly data remain in `data/raw/`. Extract the archive into a separate folder
and copy that raw directory if reproducing the old experiment.

These results used adjusted labels (43.5 °C or 40 °C with a 3.5 °C departure),
1991–2020 normals, and an ANN whose squared-error updates were reported as
weighted cross-entropy. They are legacy results, not evidence for the corrected
model. The deeper PyTorch ensemble, fuzzy inference and GA are optional theory
extensions; their old metrics must not be compared directly with the new target.
