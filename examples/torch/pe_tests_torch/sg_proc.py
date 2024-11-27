import psutil
import os

# Function to get the current process memory usage in MB
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    return memory_usage

# Example function that allocates some memory
def allocate_memory():
    a = [i for i in range(1000000)]  # Allocate a list with 1 million integers
    return a

if __name__ == "__main__":
    print(f"Memory usage before allocation: {get_memory_usage():.2f} MB")
    allocated_memory = allocate_memory()
    print(f"Memory usage after allocation: {get_memory_usage():.2f} MB")

    # Cleanup
    del allocated_memory
    print(f"Memory usage after deallocation: {get_memory_usage():.2f} MB")
