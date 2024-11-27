class IOCostOptimizer:
    """ A class for estimating and optimizing I/O costs with advanced features. """

    def __init__(self, base_read_cost, base_write_cost):
        """ Initialize with base costs for read and write operations. """
        self.base_read_cost = base_read_cost
        self.base_write_cost = base_write_cost
        self.dynamic_read_cost = base_read_cost
        self.dynamic_write_cost = base_write_cost
        self.load_factor = 1.0
        self.io_log = []

    def adjust_dynamic_cost(self, current_load):
        """ Adjust read and write costs based on the current system load. """
        self.load_factor = max(1.0, current_load)  # Load factor shouldn't be less than 1
        self.dynamic_read_cost = self.base_read_cost * self.load_factor
        self.dynamic_write_cost = self.base_write_cost * self.load_factor

    def estimate_query_cost(self, read_operations, write_operations):
        """ Estimate the cost of a query based on read and write operations. """
        cost = (read_operations * self.dynamic_read_cost + 
                write_operations * self.dynamic_write_cost)
        self.log_io_operation(read_operations, write_operations, cost)
        return cost

    def optimize_query(self, query, context):
        """ Optimize a given query based on the context. """
        # Optimization logic can be based on context
        optimized_query = query  # This is a placeholder for real optimization logic
        if context == 'high_load':
            optimized_query = self.modify_query_for_load(optimized_query)
        elif context == 'low_cost':
            optimized_query = self.modify_query_for_cost(optimized_query)
        return optimized_query

    def modify_query_for_load(self, query):
        # Logic to modify the query for high load situations
        return query

    def modify_query_for_cost(self, query):
        # Logic to modify the query for cost efficiency
        return query

    def log_io_operation(self, read_ops, write_ops, cost):
        """ Log the I/O operation for analysis. """
        self.io_log.append({'read_ops': read_ops, 'write_ops': write_ops, 'cost': cost})

    def get_io_log(self):
        """ Return the log of I/O operations. """
        return self.io_log

# Example usage
optimizer = IOCostOptimizer(base_read_cost=0.05, base_write_cost=0.1)
optimizer.adjust_dynamic_cost(current_load=1.5)  # Adjusting based on load
cost = optimizer.estimate_query_cost(read_operations=10, write_operations=5)
print(f"Estimated I/O cost: {cost}")
optimized_query = optimizer.optimize_query("SELECT * FROM table", context='high_load')
