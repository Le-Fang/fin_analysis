import csv
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import random
import json
from summarizer import Summarizer
import re

class ArticleExtractor:
    """
    A class to extract articles from a CSV file containing URLs.
    The CSV file should have two columns: website_name and url.
    The website_name is used to determine the extraction method.
    The class supports parallel processing using threads.
    It also includes a summarization feature using a language model.
    """

    def __init__(self, csv_path, max_workers=10, rate_limit=1.0):
        """
        :param csv_path: Path to the CSV file containing URLs.
        :param max_workers: Number of threads for parallel processing.
        :param rate_limit: Minimum seconds between requests per thread.
        """
        self.csv_path = csv_path
        self.max_workers = max_workers
        self.rate_limit = rate_limit  # seconds

        # List to store extracted articles
        # Each article will be a dictionary with keys: url, title, content, summary
        self.articles = []

        # a LLM model for summarization
        # You can replace this with your own summarization model
        self.summarizer = Summarizer()

    def read_csv(self):
        """Read URLs from the CSV file."""
        # Assuming the CSV file has two columns: website_name and url
        items = []
        with open(self.csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and len(row) == 2:
                    website_name = row[0].strip().lower()
                    url = row[1].strip()
                    if website_name and url:
                        items.append((website_name, url))
        return items

    def fetch_and_extract(self, website_name, url):
        """Fetch the webpage and extract the article."""
        if website_name == "wikipedia":
            return self.extract_wiki_article(url)
        elif website_name == "investopedia":
            return self.extract_investopedia_article(url)
        else:
            # Default to generic article extraction
            # You can add more specific handlers for other websites here
            return self.extract_article(url)
    
    def process(self):
        """Process all URLs in parallel and collect results."""
        items = self.read_csv()
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_and_extract, website_name, url): url for (website_name, url) in items}
            for future in as_completed(future_to_url):
                result = future.result()
                results.append(result)
        self.articles = results
        return results

    def save_to_csv(self, output_path):
        """Save the extracted articles to a CSV file."""
        fieldnames = ['url', 'title', 'content', 'summary', 'error']
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in self.articles:
                writer.writerow(article)

    def save_to_json(self, output_path):
        """Save the extracted articles to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.articles, jsonfile, ensure_ascii=False, indent=2)

    def soup_clean_up(self, soup):
        """Clean up the BeautifulSoup object by removing unwanted elements."""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Remove Wikipedia-specific clutter
        for element in soup.find_all(class_=['navbox', 'infobox', 'sidebar', 'metadata']):
            element.decompose()

        # Remove elements with common unwanted IDs
        for element in soup.find_all(id=['toc', 'catlinks', 'external-links', 'references']):
            element.decompose()

        # Remove citation links like [1], [2], etc.
        for sup in soup.find_all('sup', class_='reference'):
            sup.decompose()

        return
    
    def extract_article(self, url):
        """Extract article content from a generic URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ArticleExtractor/1.0)"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Clean up the soup object
            self.soup_clean_up(soup)

            # Extract title
            title = soup.title.string.strip() if soup.title else "No Title Found"

            # Rule-based content extraction
            content = ""
            
            # Try common article content selectors in order of priority
            content_selectors = [
                'article',
                '[role="main"]',
                'main',
                '.content',
                '.article-content',
                '.post-content',
                '#content',
                '.entry-content',
                '.article-body'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text(separator='\n', strip=True)
                    break
            
            # Fallback: extract from body and filter paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

            # Generate a summary if content is available
            summary = self.summarizer.get_summary(content) if content else None

            # Respect rate limiting
            sleep(self.rate_limit + random.uniform(0, 0.5))

            return {
                'url': url,
                'title': title,
                'content': content,
                'summary': summary
            }

        except Exception as e:
            return {
                'url': url,
                'title': None,
                'content': None,
                'summary': None,
                'error': str(e)
            }

    def extract_wiki_article(self, url):
        """Extract article content from a Wikipedia URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ArticleExtractor/1.0)"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Clean up the soup object
            self.soup_clean_up(soup)

            title = soup.title.string.strip() if soup.title else "No Title Found"

            # Extract main content (look for a specific ID)
            content_element = soup.find(id="mw-content-text")
            content = content_element.get_text(separator='\n', strip=True)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

            # Generate a summary
            summary = self.summarizer.get_summary(content)

            # Respect rate limiting
            sleep(self.rate_limit + random.uniform(0, 0.5))

            return {
                'url': url,
                'title': title,
                'content': content,
                'summary': summary
            }

        except Exception as e:
            return {
                'url': url,
                'title': None,
                'content': None,
                'summary': None,
                'error': str(e)
            }

    def extract_investopedia_article(self, url):
        """Extract article content from an Investopedia URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ArticleExtractor/1.0)"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Clean up the soup object
            self.soup_clean_up(soup)

            # Extract title
            title = soup.title.string.strip() if soup.title else "No Title Found"

            # Extract main content (look for a specific ID)
            # This is specific to the Investopedia article structure
            main_content_element = soup.find(id="mntl-sc-page_1-0")
            content = main_content_element.get_text(separator='\n', strip=True)
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

            # Generate a summary
            summary = self.summarizer.get_summary(content)

            # Respect rate limiting
            sleep(self.rate_limit + random.uniform(0, 0.5))

            return {
                'url': url,
                'title': title,
                'content': content,
                'summary': summary
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': None,
                'content': None,
                'summary': None,
                'error': str(e)
            }


if __name__ == "__main__":
    # instantiate the extractor
    extractor = ArticleExtractor(csv_path='data/FinCatch_Sources_Medium.csv', max_workers=10, rate_limit=1.0)
    
    # process the CSV file and extract articles
    extractor.process()

    # save the extracted articles to a CSV file
    extractor.save_to_json('data/extracted_articles.json')
