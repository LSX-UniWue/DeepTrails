from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

MAX_SEQ_LEN = 200
WEIGHTS = [2**-i for i in range(0, MAX_SEQ_LEN)]


def prepare_inputs(
    data: list[list[int]] | list[list[dict[str, Any]]],
    vocab_size: int,
    walk_type: str = None,
    args: dict = None,
    window_size: int = MAX_SEQ_LEN,
):
    contexts = []
    targets: list[int] = []
    indices: list[int] = []

    if walk_type is not None:
        annotated_walks = data
        index = 1 if walk_type == "fkt" else 2
        data = [[args[f"{walk_type}2id"][w[index]] for w in walk] for walk in annotated_walks]
        vocab_size = max(args[f"{walk_type}2id"].values()) + 1

    for sequence in data:
        sequence = [0] + [tok + 2 for tok in sequence] + [1]
        for i in range(1, len(sequence)):
            contexts.append(sequence[max(i - window_size, 0) : i])
            targets.append(sequence[i])
            indices.append(i)
    contexts = _weight_history(data=contexts, vocab_size=vocab_size)

    new_contexts = []
    if walk_type is not None:
        current_annotated_walk = None
        current_annotated_walk_idx = 0
        for i, (c, idx) in enumerate(zip(contexts, indices)):
            if idx == 1:
                current_annotated_walk = annotated_walks[current_annotated_walk_idx]
                current_annotated_walk_idx += 1
                feature_obj = current_annotated_walk[0]
            # concatenate feature vector to context vector
            new_contexts.append(np.concatenate((c, list(feature_obj[-1].values()))))
        contexts = new_contexts

    return contexts, targets, indices


def _weight_history(data: list[int], vocab_size: int):
    # Transform contexts to a matrix representation
    X = np.zeros((len(data), vocab_size + 2))
    # with weights, so that the last character has the highest weight
    for i, context in enumerate(data):
        for j, token in enumerate(reversed(context)):
            X[i, token] += WEIGHTS[j]
    return X


def generate(
    model: RandomForestClassifier,
    initial_context: list[int],
    features: dict[str, Any],
    max_length: int,
):
    text = initial_context
    for _ in range(max_length):
        # Transform the current context into the matrix representation
        X = _weight_history(
            data=[[state for state in text[-MAX_SEQ_LEN:]]],
            vocab_size=model.n_features_in_ - 2 - len(features),
        )
        # Add the feature vector to the context vector
        assert len(features) == len(model.feature_ordering)
        features = {model.feature_ordering[i]: val for i, val in enumerate(features.values())}
        X = [np.concatenate((X.squeeze(), list(features.values())))]
        # Predict the next character
        prediction = model.predict(X)
        # Decode the prediction
        next_char = [int(prediction)]
        # Add the predicted character to the text
        text += next_char
        if text[-1] == 1:
            break
    return text


def get_probabilities_per_token(model, inputs: list[float], targets: list[int]):
    probabilities = []
    predictions = model.predict_proba(inputs)
    for prediction, target in zip(predictions, targets):
        probabilities.append(prediction[model.classes_ == target][0])
    return probabilities
