# pytest config for the ch19 companion tests
def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a real CUDA device (graph capture)")


def pytest_collection_modifyitems(config, items):
    import torch

    if torch.cuda.is_available():
        return
    skip_cuda = __import__("pytest").mark.skip(reason="no CUDA device on this host")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
