import random

def label_encoding(inputdata):
    encoding_map_dictionary = {}
    encoded_data = []

    random.seed(123)  # Set a seed for reproducibility

    for obj in inputdata:
        if obj not in encoding_map_dictionary:
            encoding_map_dictionary[obj] = int(random.random() * 1000)  # Map random float to integer
        encoded_data.append(encoding_map_dictionary[obj])
    
    return encoded_data
def read_dataset(filename):
  dataset = []
  with open(filename,'r') as file:
    lines = file.readlines()
    for line in lines[1:]:
      row = line.strip().split(',')
      dataset.append(row)
  return dataset


 


def main():
  filename = "/Users/Dell/Desktop/sem-4/ML/Iris.csv"
  penguin_dataset = read_dataset(filename)



  species_column = [row[5] for row in penguin_dataset]

  encoded_species = label_encoding(species_column)

  print("Original species data:", species_column)
  print("Encoded species data:", encoded_species)


if __name__ == "__main__":
  main()
