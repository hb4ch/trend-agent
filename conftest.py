import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep default pytest runs hermetic unless the caller opts into marker filtering."""
    if config.option.markexpr:
        return

    deselected = []
    selected = []
    for item in items:
        if item.get_closest_marker("live_llm") or item.get_closest_marker("integration"):
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
