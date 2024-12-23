class indexinfo:
    def __init__(self, index_id, name, outlier,
                 max_normal_value=0, min_normal_value=0, max_outlier=0, min_outlier=0):
        self.id = index_id
        self.name = name
        self.outlier = outlier
        self.max_normal_value = max_normal_value
        self.min_normal_value = min_normal_value
        self.max_outlier = max_outlier
        self.min_outlier = min_outlier
