import importlib


def get_loader(dataset_name):
    return getattr(importlib.import_module(f'BOCFRLM.dataset.{dataset_name}'), 'DataLoader')
