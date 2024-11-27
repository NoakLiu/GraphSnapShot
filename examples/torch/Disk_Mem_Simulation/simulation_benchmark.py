import time

class DiskMemoryBenchmark:
    def __init__(self, disk_read_time, disk_write_time, memory_access_time):
        """
        Initialize the benchmark with simulated times for disk read, disk write, and memory access.
        :param disk_read_time: Simulated time to read from disk (in seconds)
        :param disk_write_time: Simulated time to write to disk (in seconds)
        :param memory_access_time: Simulated time for memory access (in seconds)
        """
        self.disk_read_time = disk_read_time
        self.disk_write_time = disk_write_time
        self.memory_access_time = memory_access_time

    def simulate_disk_read(self):
        """ Simulate reading from disk """
        start_time = time.time()
        time.sleep(self.disk_read_time)  # Simulate disk read delay
        return time.time() - start_time

    def simulate_disk_write(self):
        """ Simulate writing to disk """
        start_time = time.time()
        time.sleep(self.disk_write_time)  # Simulate disk write delay
        return time.time() - start_time

    def simulate_memory_access(self):
        """ Simulate accessing memory """
        start_time = time.time()
        time.sleep(self.memory_access_time)  # Simulate memory access delay
        return time.time() - start_time

# Example usage
benchmark = DiskMemoryBenchmark(disk_read_time=5, disk_write_time=1, memory_access_time=0.0001)

disk_read_duration = benchmark.simulate_disk_read()
disk_write_duration = benchmark.simulate_disk_write()
memory_access_duration = benchmark.simulate_memory_access()

print(f"Simulated Disk Read Duration: {disk_read_duration} seconds")
print(f"Simulated Disk Write Duration: {disk_write_duration} seconds")
print(f"Simulated Memory Access Duration: {memory_access_duration} seconds")
