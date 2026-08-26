# pytest config for the ch17 companion tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: real-multiprocessing end-to-end (spawned WorkerProc children)"
    )
