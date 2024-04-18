import os
from haystack import component
from haystack.dataclasses import Document
from browserbase import Browserbase
from typing import List


@component
class BrowserbaseFetcher:
    def __init__(self, api_key: str = os.environ["BROWSERBASE_KEY"]) -> None:
        self.browserbase = Browserbase(api_key=api_key)

    @component.output_types(documents=List[Document])
    def run(self, urls: List[str], text_content: bool = False):
        pages = self.browserbase.load_urls(urls, text_content)

        documents = []
        for i, page in enumerate(pages):
            documents.append(
                Document(
                    content=page,
                    meta={
                        "url": urls[i],
                    },
                )
            )

        return {"documents": documents}
