# utevsky2016

Clean up utevsky's monkey experiment neuron dataset and run variational inference with it.

# Notebooks
A quick rundown of notebooks and what's in them:

- `validate_edward_model_linear_regression`: this one works. This shows that data generated from the Poisson model with log-linear firing rate A + BX (X known regressors) can correctly recover A and B.
- `validate_edward_model_manual_hierarchy`: my best (failed) effort to recover latents in Poisson model with log-linear rate A + BX + CZ. My suspicion is that the matrix CZ can be factorized in too many ways, with too many local minima. If Z were real-valued instead of binary, this would be more like a factor model, where (I think) better results might be had (though they'd only be good up to a rotation in factor space).

  This model does exact coordinate ascent updates for Z and pi. It also has a hierarchical prior over A, B, and C.
- `validate_edward_model_manual_latent updates`: same as `manual_hierarchy` above but without the hierarchy part

The following notebooks are deprecated and will be removed:

- `validate_edward_model_multiple_inference_steps`: where I realized that it was best not to update all the variables at the same time, but to do some alternation between related groups of them.
- `validate_edward_model_manual_hierarchy_all_grad`: gradient updates for everything, including Z

- `validate_edward_model`: first try
- `validate_edward_regression_model`: another early try
