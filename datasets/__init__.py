class Dataset(list):
    def __init__(self, data=None, features=None):
        super().__init__(data or [])
        self.features = features

    @classmethod
    def from_dict(cls, mapping, features=None):
        ds = cls([], features)
        ds.data = mapping
        return ds

    @classmethod
    def from_pandas(cls, df, features=None):
        ds = cls([], features)
        ds._df = df
        return ds

    def set_transform(self, fn):
        self._transform = fn

    def to_pandas(self):
        import pandas as pd
        return getattr(self, '_df', pd.DataFrame(self.data))

def load_dataset(*a, **k):
    return Dataset.from_dict({})
