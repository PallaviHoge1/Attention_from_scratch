from attention import attention

# Define the query, keys, and values
query = [1, 0, 1]
keys = [
    [1, 0, 0], 
    [0, 1, 1], 
    [1, 1, 0]
]
values = [
    [5, 10, 15], 
    [20, 25, 30], 
    [35, 40, 45]
]

# Calculate the attention result
result = attention(query, keys, values)

# Print the result
print("Attention Output:", result)
