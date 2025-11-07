def test_import_package():
    """Simple smoke test: package imports and has expected symbol."""
    import hossam

    assert hasattr(hossam, 'load_data')
