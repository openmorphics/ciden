def test_import():
    import sr_ciden  # noqa: F401


def test_version():
    import sr_ciden
    assert sr_ciden.__version__ == "0.1.0"
