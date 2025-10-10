from montecarlo_nystrom.main import add


def test_add():
    """Adding two number works as expected."""
    assert add(1, 1) == 2
