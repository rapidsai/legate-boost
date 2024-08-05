import legateboost


def test_version_constants_are_populated():
    # __version__ should always be non-empty
    assert isinstance(legateboost.__version__, str)
    assert len(legateboost.__version__) > 0
