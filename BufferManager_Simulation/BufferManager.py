class BufferManager:
    """ A simple buffer manager that handles data storage and retrieval. """

    def __init__(self, capacity):
        """ Initialize the buffer manager with a specified capacity. """
        self.capacity = capacity
        self.buffer = {}
        self.access_order = []

    def load_data(self, key, data):
        """ Load data into the buffer. """
        if key not in self.buffer:
            if len(self.buffer) >= self.capacity:
                oldest_key = self.access_order.pop(0)
                del self.buffer[oldest_key]
            self.buffer[key] = data
            self.access_order.append(key)

    def get_data(self, key):
        """ Retrieve data from the buffer. """
        if key in self.buffer:
            data = self.buffer[key]
            self.access_order.remove(key)
            self.access_order.append(key)
            return data
        else:
            return None

    def store_data(self, key, data):
        """ Store data in the buffer. """
        self.load_data(key, data)
