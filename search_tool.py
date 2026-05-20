from deep_researcher import search_backend


def search(query: str) -> str:
    return search_backend(query)


__all__ = ["search"]
