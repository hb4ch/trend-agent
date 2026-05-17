import pytest

import llm_provider


def test_llm_provider_has_two_normalized_tiers():
    heavy_aliases = ["heavy", "main", "driver", "deepseek", "deepseek-v4-pro"]
    light_aliases = ["light", "gemma", "matching", "labeling", "sentiment", "sentiment_matching"]

    for alias in heavy_aliases:
        assert llm_provider.normalize_model_tier(alias) == "heavy"

    for alias in light_aliases:
        assert llm_provider.normalize_model_tier(alias) == "light"


def test_llm_provider_rejects_removed_chat_tiers():
    with pytest.raises(ValueError, match="Unknown LLM tier"):
        llm_provider.normalize_model_tier("zhipu")


def test_light_tier_forces_gemma_sampling_params(monkeypatch):
    monkeypatch.setattr(llm_provider, "GEMMA_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(llm_provider, "GEMMA_API_KEY", "dummy")
    provider = llm_provider.LLMProvider()

    llm = provider.get_llm("light", temperature=0.1)

    assert llm.temperature == 1.0
    assert llm.top_p == 0.95
    assert llm.extra_body == {"top_k": 64}
