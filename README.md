# Browserbase Haystack Fetcher

[Browserbase](https://browserbase.com) is a developer platform to reliably run, manage, and monitor headless browsers.

Power your AI data retrievals with:
- [Serverless Infrastructure](https://docs.browserbase.com/under-the-hood) providing reliable browsers to extract data from complex UIs
- [Stealth Mode](https://docs.browserbase.com/features/stealth-mode) with included fingerprinting tactics and automatic captcha solving
- [Session Debugger](https://docs.browserbase.com/features/sessions) to inspect your Browser Session with networks timeline and logs
- [Live Debug](https://docs.browserbase.com/guides/session-debug-connection/browser-remote-control) to quickly debug your automation

## Installation and setup

- Get an API key from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`).
- Install the required dependencies:

```
pip install browserbase-haystack
```

## Usage

You can load webpages into Haystack using `BrowserbaseFetcher`. Optionally, you can set `text_content` parameter to convert the pages to text-only representation.

### Standalone

```py
from browserbase_haystack import BrowserbaseFetcher

browserbase_fetcher = BrowserbaseFetcher()
browserbase_fetcher.run(urls=["https://example.com"], text_content=False)
```

### In a pipeline

```py
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from browserbase_haystack import BrowserbaseFetcher

prompt_template = (
    "Tell me the titles of the given pages. Pages: {{ documents }}"
)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator()

browserbase_fetcher = BrowserbaseFetcher()

pipe = Pipeline()
pipe.add_component("fetcher", browserbase_fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("fetcher.documents", "prompt_builder.documents")
pipe.connect("prompt_builder.prompt", "llm.prompt")
result = pipe.run(data={"fetcher": {"urls": ["https://example.com"]}})
```
