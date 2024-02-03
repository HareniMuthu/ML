# Function to perform matrix multiplication
def matrix_multiply(A, B):
    # Assuming A and B are 2D matrices
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Incompatible matrix dimensions for multiplication.")

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Function to calculate matrix power without using numpy
def matrix_power(A, m):
    # Check if the input matrix is square
    if not is_square_matrix(A):
        raise ValueError("Input matrix must be square for matrix power calculation.")

    # Initialize result as an identity matrix
    result = [[1 if i == j else 0 for j in range(len(A))] for i in range(len(A))]

    # Multiply A with itself m times
    for _ in range(m):
        result = matrix_multiply(result, A)

    return result

# Function to check if a matrix is square
def is_square_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    return rows == cols

# Function to find the most occurring alphabet character in a string
def highest_occuring_character(input_string):
    char_count = {}
    for char in input_string:
        if char.isalpha():
            char_count[char] = char_count.get(char, 0) + 1

    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]

    return f"Most frequent character: {max_char}, Occurrence count: {max_count}"

# Example Usage:
matrix_A = [[1, 2], [3, 4]]
power_m = 2
result_power = matrix_power(matrix_A, power_m)
print(f"A raised to the power of {power_m}:\n{result_power}")

input_str_occurrence = "abracadabra"
result_occurrence = highest_occuring_character(input_str_occurrence)
print(result_occurrence)
