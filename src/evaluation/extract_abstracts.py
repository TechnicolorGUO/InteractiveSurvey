import os
import feedparser
import urllib.parse

def get_arxiv_abstract_by_title(title):
    query_title = urllib.parse.quote(f'"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query=ti:{query_title}&max_results=1"
    feed = feedparser.parse(url)
    if feed.entries:
        return feed.entries[0].summary
    return "No abstract found"

def main():
    d = os.path.dirname(os.path.abspath(__file__))
    inp = os.path.join(d, 'arxiv_downloads')
    out = os.path.join(d, 'abstract_survey')
    if not os.path.exists(out):
        os.makedirs(out)
    for folder_name in os.listdir(inp):
        fp = os.path.join(inp, folder_name)
        if not os.path.isdir(fp):
            continue
        md = os.path.join(out, f"{folder_name}.md")
        data = []
        pdfs = sorted([x for x in os.listdir(fp) if x.lower().endswith('.pdf')])
        for i, pdf in enumerate(pdfs, 1):
            title = os.path.splitext(pdf)[0]
            abs_text = get_arxiv_abstract_by_title(title)
            data.append(f"{i}: {pdf}\n{abs_text}\n")
        with open(md, 'w', encoding='utf-8') as f:
            f.write(f"# {folder_name} Abstracts\n\n")
            for line in data:
                f.write(line + "\n")

if __name__ == "__main__":
    main()
