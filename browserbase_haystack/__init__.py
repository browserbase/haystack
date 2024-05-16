from haystack import component
from haystack.dataclasses import Document
from browserbase import Browserbase
from typing import Optional, List, Sequence


@component
class BrowserbaseFetcher:
    def __init__(
        self, api_key: Optional[str] = None, project_id: Optional[str] = None
    ) -> None:
        self.browserbase = Browserbase(api_key=api_key, project_id=project_id)

    @component.output_types(documents=List[Document])
    def run(
        self,
        urls: Sequence[str],
        text_content: bool = False,
        session_id: Optional[str] = None,
        proxy: Optional[bool] = None,
    ):
        pages = self.browserbase.load_urls(
            urls,
            text_content,
            session_id,
            proxy,
        )

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
