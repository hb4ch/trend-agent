import json
import os
import unittest


class _FakeWebSearch:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def web_search(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.web_search = _FakeWebSearch(response)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    @property
    def __dict__(self):
        return self.payload


class TestZhipuRawSearch(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("ZHIPUAI_API_KEY", "test-key")
        os.environ["DEBUG_ZHIPU_SEARCH"] = "0"
        os.environ["DEBUG_DEEPSEEK"] = "0"
        os.environ["DEBUG_SILICONFLOW"] = "0"

    def test_returns_structured_results_and_parses_site_filter(self):
        from deep_researcher import ZhipuSearchTool

        fake_response = _FakeResponse(
            {
                "data": [
                    {
                        "title": "测试标题",
                        "url": "http://example.com/a",
                        "snippet": "测试摘要",
                        "date": "2026-01-01",
                    }
                ]
            }
        )
        tool = ZhipuSearchTool()
        tool.client = _FakeClient(fake_response)

        out = tool.search("site:cninfo.com.cn 维力医疗 重大合同")
        payload = json.loads(out)

        self.assertIn("results", payload)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["url"], "http://example.com/a")
        self.assertIn("summary", payload)

        self.assertEqual(len(tool.client.web_search.calls), 1)
        call = tool.client.web_search.calls[0]
        self.assertEqual(call["search_domain_filter"], "cninfo.com.cn")
        self.assertNotIn("site:cninfo.com.cn", call["search_query"])

    def test_no_results_returns_empty_list_and_summary(self):
        from deep_researcher import ZhipuSearchTool

        fake_response = _FakeResponse({"data": []})
        tool = ZhipuSearchTool()
        tool.client = _FakeClient(fake_response)

        out = tool.search("不存在的关键词")
        payload = json.loads(out)

        self.assertEqual(payload["results"], [])
        self.assertTrue(isinstance(payload["summary"], str))
        self.assertNotEqual(payload["summary"], "")


if __name__ == "__main__":
    unittest.main()
