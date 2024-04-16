import time
import threading
import random

class DiskMemoryBenchmark:
    def __init__(self, disk_read_time, disk_write_time, memory_access_time):
        self.disk_read_time = disk_read_time
        self.disk_write_time = disk_write_time
        self.memory_access_time = memory_access_time
        self.cache = {}  # Simple cache for demonstration
        
    def simulate_disk_read(self, file_size=1):
        """ Simulate reading from disk with varying file size """
        start_time = time.time()
        time.sleep(self.disk_read_time * file_size)
        return time.time() - start_time

    def simulate_disk_write(self, file_size=1):
        """ Simulate writing to disk with varying file size """
        start_time = time.time()
        time.sleep(self.disk_write_time * file_size)
        return time.time() - start_time

    def simulate_memory_access(self):
        """ Simulate accessing memory """
        start_time = time.time()
        time.sleep(self.memory_access_time)
        return time.time() - start_time

    def simulate_with_threads(self, operation, num_threads=2, args=()):
        """ Simulate operations using multiple threads """
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=operation, args=args)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def simulate_cached_read(self, file_id):
        """ Simulate reading with a simple cache mechanism """
        if file_id in self.cache:
            return 0  # Instant access from cache
        else:
            duration = self.simulate_disk_read()
            self.cache[file_id] = True  # Simulate caching the file
            return duration

# Example usage
benchmark = DiskMemoryBenchmark(disk_read_time=0.05, disk_write_time=0.02, memory_access_time=0.0001)

# Multi-threaded disk read simulation
benchmark.simulate_with_threads(benchmark.simulate_disk_read, num_threads=4, args=(2,))

# Simulating cached read
file_id = random.randint(1, 100)  # Random file ID
first_read_duration = benchmark.simulate_cached_read(file_id)
second_read_duration = benchmark.simulate_cached_read(file_id)  # Should be faster due to caching

print(f"First Read Duration (possibly uncached): {first_read_duration} seconds")
print(f"Second Read Duration (cached): {second_read_duration} seconds")

disk_read_duration = benchmark.simulate_disk_read()
disk_write_duration = benchmark.simulate_disk_write()
memory_access_duration = benchmark.simulate_memory_access()

print(f"Simulated Disk Read Duration: {disk_read_duration} seconds")
print(f"Simulated Disk Write Duration: {disk_write_duration} seconds")
print(f"Simulated Memory Access Duration: {memory_access_duration} seconds")
