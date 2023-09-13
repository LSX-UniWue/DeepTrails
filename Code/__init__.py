
MIXTURES = {
    "even": {"even": 1.0, "odd": 0.0},
    "odd": {"even": 0.0, "odd": 1.0},
    "first-even": {0.5: {"even": 1.0, "odd": 0.0}, 1.0: {"even": 0.0, "odd": 1.0}},
    "first-odd": {0.5: {"even": 0.0, "odd": 1.0}, 1.0: {"even": 1.0, "odd": 0.0}},
    "three-even-one-odd": {"even": 3, "odd": 1},
    "three-odd-one-even": {"even": 1, "odd": 3},
    "two-even-two-odd": {"even": 2, "odd": 2},
    "rand": {"even": 0.5, "odd": 0.5},
    "tele": {"teleport": 1.0, "even": 0.0, "odd": 0.0},
    "even-biased": {"even": 0.9, "odd": 0.1},
    "odd-biased": {"even": 0.1, "odd": 0.9},
    "first-even-biased": {0.5: {"even": 0.9, "odd": 0.1}, 1.0: {"even": 0.1, "odd": 0.9}},
    "first-odd-biased": {0.5: {"even": 0.1, "odd": 0.9}, 1.0: {"even": 0.9, "odd": 0.1}},
    "rand-biased": {"teleport": 0.1, "even": 0.45, "odd": 0.45},
    "tele-biased": {"teleport": 0.9, "even": 0.05, "odd": 0.05},
}