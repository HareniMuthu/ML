import math

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must have the same dimension")
    
    sum_of_squared_diff = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(sum_of_squared_diff)

def manhattan_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must have the same dimension")
    
    return sum(abs(x - y) for x, y in zip(vector1, vector2))

def get_vector_input():
    vector_str = input("Enter the vector (comma-separated values): ")
    vector = [float(x.strip()) for x in vector_str.split(',')]
    return vector

def main():
    print("Enter the first vector:")
    vector1 = get_vector_input()
    
    print("Enter the second vector:")
    vector2 = get_vector_input()
    
    try:
        euclidean_dist = euclidean_distance(vector1, vector2)
        manhattan_dist = manhattan_distance(vector1, vector2)
        print("Euclidean distance:", euclidean_dist)
        print("Manhattan distance:", manhattan_dist)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
