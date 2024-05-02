import xml.etree.ElementTree as ET
from settings import PROJECT_ROOT_DIR


def parse_wikipedia_abstracts(file_path: str) -> list[str]:
    context = ET.iterparse(file_path, events=("start", "end"))
    context = iter(context)
    event, root = next(context)

    texts: list[str] = []
    for event, elem in context:
        if event == "end" and elem.tag == "doc":
            title = elem.find("title").text
            url = elem.find("url").text
            abstract = elem.find("abstract").text
            texts.append(
                f"""
                ##{title}
                URL: {url}
                Abstract: {abstract}    
            """.strip()
            )
            root.clear()
    return texts


texts = parse_wikipedia_abstracts(
    PROJECT_ROOT_DIR / "train" / "example_dataset" / "enwiki-latest-abstract-small.xml"
)
