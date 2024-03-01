import psutil
import time

def memory_usage_tester(func):
    """
    Test memory usage of a function.

    Args:
    - func: The function to be tested.

    Returns:
    - memory_info: A dictionary containing memory usage information.
    """
    # Get memory usage before running the function
    start_memory = psutil.Process().memory_info().rss

    # Run the function
    func()

    # Get memory usage after running the function
    end_memory = psutil.Process().memory_info().rss

    # Calculate memory usage increment
    memory_usage = end_memory - start_memory

    # Convert memory usage to appropriate unit
    memory_usage_kb = memory_usage / 1024
    memory_usage_mb = memory_usage_kb / 1024
    memory_usage_gb = memory_usage_mb / 1024

    # Construct a dictionary with memory usage information
    memory_info = {
        "start_memory": start_memory,
        "end_memory": end_memory,
        "memory_usage_bytes": memory_usage,
        "memory_usage_kb": memory_usage_kb,
        "memory_usage_mb": memory_usage_mb,
        "memory_usage_gb": memory_usage_gb
    }

    return memory_info

def create_large_list():
    large_list = [i for i in range(1000000)]  # Generate a list with 1 million elements
    time.sleep(1)  # Simulate function runtime
    return large_list

def simple_loop():
    for i in range(1000000):  # A simple loop
        pass
    time.sleep(1)  # Simulate function runtime

def simple_string_concatenation():
    result = ""
    for i in range(10000):  # Repeat concatenation 10,000 times
        result += "a" * 1000  # Concatenate 1000 characters each time
    time.sleep(1)  # Simulate function runtime

if __name__ == "__main__":
    # Test memory usage for create_large_list function
    print("Memory usage for create_large_list:")
    memory_info = memory_usage_tester(create_large_list)
    print(memory_info)

    # Test memory usage for simple_loop function
    print("\nMemory usage for simple_loop:")
    memory_info = memory_usage_tester(simple_loop)
    print(memory_info)

    # Test memory usage for simple_string_concatenation function
    print("\nMemory usage for simple_string_concatenation:")
    memory_info = memory_usage_tester(simple_string_concatenation)
    print(memory_info)
