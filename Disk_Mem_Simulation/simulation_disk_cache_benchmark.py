import time

class DiskCacheBenchmark:
    def __init__(self, disk_read_time_per_unit, disk_write_time_per_unit, cache_access_time_per_unit):
        """
        Initialize the benchmark with simulated times per unit for disk read, disk write, and cache access.
        :param disk_read_time_per_unit: Simulated time to read from disk per unit (in seconds)
        :param disk_write_time_per_unit: Simulated time to write to disk per unit (in seconds)
        :param cache_access_time_per_unit: Simulated time for cache access per unit (in seconds)
        """
        self.disk_read_time_per_unit = disk_read_time_per_unit
        self.disk_write_time_per_unit = disk_write_time_per_unit
        self.cache_access_time_per_unit = cache_access_time_per_unit

    def simulate_disk_read(self, amount):
        """ Simulate reading from disk """
        start_time = time.time()
        time.sleep(self.disk_read_time_per_unit * amount)  # Simulate disk read delay
        return time.time() - start_time

    def simulate_disk_write(self, amount):
        """ Simulate writing to disk """
        start_time = time.time()
        time.sleep(self.disk_write_time_per_unit * amount)  # Simulate disk write delay
        return time.time() - start_time

    def simulate_cache_access(self, amount):
        """ Simulate accessing cache """
        start_time = time.time()
        time.sleep(self.cache_access_time_per_unit * amount)  # Simulate cache access delay
        return time.time() - start_time

# Example usage
benchmark = DiskCacheBenchmark(disk_read_time_per_unit=0.005, disk_write_time_per_unit=0.001, cache_access_time_per_unit=0.000001)

disk_read_duration = benchmark.simulate_disk_read(1000)  # Simulate reading 1000 units
disk_write_duration = benchmark.simulate_disk_write(500)  # Simulate writing 500 units
cache_access_duration = benchmark.simulate_cache_access(2000)  # Simulate accessing cache for 2000 units

print(f"Simulated Disk Read Duration: {disk_read_duration} seconds")
print(f"Simulated Disk Write Duration: {disk_write_duration} seconds")
print(f"Simulated Cache Access Duration: {cache_access_duration} seconds")
