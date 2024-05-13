import unittest
from browserbase_haystack import BrowserbaseFetcher
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder


class BrowserbaseFetcherTestCase(unittest.TestCase):
    def setUp(self):
        self.browserbase_fetcher = BrowserbaseFetcher()

    def test_browserbase_fetcher(self):
        result = self.browserbase_fetcher.run(urls=["https://example.com"])
        self.assertIn("Example Domain", result["documents"][0].content)

    def test_browserbase_fetcher_in_pipeline(self):
        prompt_template = (
            "Tell me the titles of the given pages. Pages: {{ documents }}"
        )
        prompt_builder = PromptBuilder(template=prompt_template)
        llm = OpenAIGenerator()

        pipe = Pipeline()
        pipe.add_component("fetcher", self.browserbase_fetcher)
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)

        pipe.connect("fetcher.documents", "prompt_builder.documents")
        pipe.connect("prompt_builder.prompt", "llm.prompt")
        result = pipe.run(data={"fetcher": {"urls": ["https://example.com"]}})
        self.assertIn("1. Example Domain", result["llm"]["replies"])
