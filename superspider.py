import os
from datetime import datetime
from trafilatura import fetch_url, extract, feeds
from trafilatura.settings import DEFAULT_CONFIG
#from trafilatura.spider import focused_crawler

os.makedirs("input", exist_ok=True)

## Set target RSS
target = "https://feeds.bbci.co.uk/news/world/rss.xml"

to_visit, known_links = [], set()

## Crawl the urls from RSS

feedurls = feeds.find_feed_urls(target)
to_visit = [link for link in feedurls if (
    ("/articles/" in link) and (link not in known_links)
    )]

## Extract and join all string together 
text = [extract(fetch_url(u))+"\n" for u in to_visit]
text = "".join(text)
## Update known_links
known_links.update(to_visit)

## Save text
text_filename = datetime.today().strftime("%Y_%m_%d_%H_%M_%S") or "index"
file_path = f'input/{text_filename}.txt' #or .md

with open(file_path, 'w', encoding='utf-8') as output:
    output.write(text)

print(f"Finish crawling. Crawled {len(to_visit)} pages")
print("Markdown saved to input/ ")