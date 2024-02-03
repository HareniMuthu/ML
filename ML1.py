import numpy as np

# Function to count pairs in the list with the given sum
def count_pairs_with_sum(lst, target_sum):
    count = 0
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] + lst[j] == target_sum:
                count += 1
    return count

# Function to calculate the range of a list
def calculate_range(lst):
    # Check if the list has less than 3 elements
    if len(lst) < 3:
        return "Range determination not possible"
    else:
        # Calculate and return the range
        return max(lst) - min(lst)

# Function to compute matrix power
def matrix_power(A, m):
    return np.linalg.matrix_power(A, m)

# Function to find the most occurring character in a string
def highest_occuring_character(input_string):
    char_count = {}
    for char in input_string:
        if char.isalpha():
            char_count[char] = char_count.get(char, 0) + 1

    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]

    return f"Most frequent character: {max_char}, Occurrence count: {max_count}"

# Example usage:

# 1. Count pairs with sum 10
lst_pairs = [2, 7, 4, 1, 3, 6]
target_sum_pairs = 10
result_pairs = count_pairs_with_sum(lst_pairs, target_sum_pairs)
print(f"Number of pairs with sum {target_sum_pairs}: {result_pairs}")

# 2. Calculate range
lst_range = [5, 3, 8, 1, 0, 4]
result_range = calculate_range(lst_range)
print(f"Calculated range: {result_range}")

# 3. Matrix power
matrix_A = np.array([[1, 2], [3, 4]])
power_m = 2
result_power = matrix_power(matrix_A, power_m)
print(f"A raised to the power of {power_m}:\n{result_power}")

# 4. Highest occurring character
input_str_occurrence = "abracadabra"
result_occurrence = highest_occuring_character(input_str_occurrence)
print(result_occurrence)
