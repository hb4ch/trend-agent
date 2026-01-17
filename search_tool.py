from deep_researcher import zhipu_search


def search(query: str) -> str:
    return zhipu_search(query)


__all__ = ["search"]
