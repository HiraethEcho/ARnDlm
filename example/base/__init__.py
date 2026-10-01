"""Row 1 of SPEC 2.3: the same training logic written against `transformers`.

`base/native/` implements the identical recipe with plain torch.nn. Two cells
that share a training loop can only differ in the model, which is what makes
the C1 logit comparison meaningful.
"""