import math

def dot_product(vector1, vector2):
    """Calculates dot product between two vectors."""
    return sum([x * y for x, y in zip(vector1, vector2)])

def softmax(scores):
    """Applies softmax to normalize the scores into probabilities."""
    exp_scores = [math.exp(score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]

def weighted_sum(weights, values):
    """Calculates the weighted sum of values based on weights."""
    weighted_values = [[w * v for v in value] for w, value in zip(weights, values)]
    return [sum(items) for items in zip(*weighted_values)]

def attention(query, keys, values):
    """Calculates attention by combining query, keys, and values."""
    # Step 1: Calculate dot product (attention scores)
    scores = [dot_product(query, key) for key in keys]

    # Step 2: Apply softmax to convert scores into probabilities (weights)
    weights = softmax(scores)

    # Step 3: Calculate weighted sum of the values
    return weighted_sum(weights, values)
