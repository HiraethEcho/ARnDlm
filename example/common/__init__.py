"""A-version shared base. Deliberately independent of the repo's root utils/.

Root utils/ is B-version territory (the part the user writes), so the A-version
cannot import from it -- otherwise every A/B comparison would be circular.
Everything the 10 cells have in common lives here.
"""