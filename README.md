# Attention Mechanisms in AI: Code Explanations

## Overview

This repository explains **Attention Mechanisms** in AI using code from scratch without relying on external libraries like NumPy. The focus is on understanding and implementing attention step-by-step using plain Python, and applying it in various AI tasks like **machine translation**, **summarization**, **image captioning**, and more.

## Code Explanations

### 1. **Dot Product Calculation**

In attention mechanisms, the **dot product** between queries and keys helps compute how "similar" or "relevant" each input is to the current focus. This forms the basis for calculating attention scores.

#### Code Snippet:

```python
def dot_product(vector1, vector2):
    return sum([x * y for x, y in zip(vector1, vector2)])
```

- `vector1` and `vector2` represent the query and key vectors.
- The `dot_product` function multiplies corresponding elements of the vectors and returns the sum of these products.

### 2. **Softmax Function**

The **Softmax** function normalizes the attention scores into probabilities, ensuring that the sum of probabilities is 1. This allows the model to focus on the most relevant parts of the input.

#### Code Snippet:

```python
import math

def softmax(scores):
    exp_scores = [math.exp(score) for score in scores]
    total = sum(exp_scores)
    return [score / total for score in exp_scores]
```

- The function takes a list of `scores` (dot products) and computes their exponential values.
- These exponentiated scores are then normalized by dividing each score by the sum of all exponentiated scores.

### 3. **Weighted Sum Calculation**

The **weighted sum** of values represents the final output of the attention mechanism. The weights (probabilities) from the softmax function are used to scale the values (which correspond to the inputs).

#### Code Snippet:

```python
def weighted_sum(weights, values):
    weighted_values = [[w * v for v in value] for w, value in zip(weights, values)]
    return [sum(items) for items in zip(*weighted_values)]
```

- `weights` are the softmax probabilities, and `values` represent the input values.
- For each value, we multiply it by its corresponding weight, and then we sum across all values to get the final output.

### 4. **Attention Mechanism**

The complete **attention mechanism** calculates attention scores, normalizes them with softmax, and generates the output using the weighted sum.

#### Code Snippet:

```python
def attention(query, keys, values):
    scores = [dot_product(query, key) for key in keys]
    weights = softmax(scores)
    return weighted_sum(weights, values)
```

- **Step 1**: The model calculates the dot product between the `query` and each `key` to get attention scores.
- **Step 2**: The attention scores are passed through the softmax function to get probabilities (weights).
- **Step 3**: The weighted sum of `values` is computed using the weights to generate the final output.

---

## Summary

This repository implements the attention mechanism using basic Python code. Each step of the attention process is explained with simple code examples, focusing on clarity and understanding of the core concepts.

---

## How to Use the Example Script

You can test the attention mechanism with the provided `example.py` script.

### Steps to run the example:

1. **Clone the repository**:
   
   ```bash
   git clone https://github.com/PallaviHoge1/Attention_from_scratch
   cd Attention_from_scratch
   ```

2. **Run the example**:

   You can execute the `example.py` script from the command line:

   ```bash
   python example.py
   ```

3. **Expected Output**:

   The script will run the attention mechanism on a sample query, keys, and values. You should see output similar to:

   ```bash
   Attention Output: [47.25, 52.25, 57.25]
   ```

   This output represents the weighted sum of the values, where each value is weighted by the attention scores computed from the query and keys.

### Understanding the Example:

In the `example.py` file:

- We define a query vector representing the word we're focusing on.
- We provide a set of keys (vectors representing the words in the input).
- We also provide a set of values (encoded information about each word).
- The `attention()` function calculates attention scores, applies softmax to get probabilities, and returns the weighted sum of the values based on these probabilities.

By running the example, you can observe how the attention mechanism processes these vectors and computes the final output.

---

Feel free to modify `example.py` to test different queries, keys, and values to explore how the attention mechanism works under various conditions.

---