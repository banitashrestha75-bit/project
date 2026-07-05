import urllib.request
from bs4 import BeautifulSoup
import json
import ssl
import os
import re
from urllib.parse import urljoin, urlparse
from logger_setup import logger

class LICTScraper:
    def __init__(self, max_pages=35):
        self.base_url = "https://lict.edu.np/"
        self.max_pages = max_pages
        self.visited = set()
        self.scraped_data = []
        
        # SSL Context to ignore certificate errors if any
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def _is_valid_url(self, url):
        """Checks if URL belongs to lict.edu.np and is not a file/resource link."""
        parsed = urlparse(url)
        # Only crawl same domain
        if parsed.netloc not in ["lict.edu.np", "www.lict.edu.np", ""]:
            return False
            
        # Skip assets and documents
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']):
            return False
            
        # Avoid dynamic and query URLs that don't represent real pages
        if parsed.query or "#" in url:
            return False
            
        return True

    def clean_text(self, text):
        """Cleans excess whitespace and newlines from scraped text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def scrape_page(self, url):
        """Scrapes a single page and extracts structured details."""
        if url in self.visited:
            return []
            
        logger.info(f"Scraping page: {url}")
        self.visited.add(url)
        
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=12) as response:
                html = response.read()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return []

        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove navigation, footer, script, and style elements to isolate main content
        for element in soup(["script", "style", "nav", "footer", "header", "aside", ".footer", ".header", ".menu", ".nav"]):
            element.decompose()

        title = soup.title.string if soup.title else "Lumbini ICT College"
        title = self.clean_text(title)
        
        # Extract headings and content
        sections = []
        current_heading = "General Information"
        current_text_blocks = []
        
        # Iterate over structural tags
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        if not main_content:
            main_content = soup
            
        for child in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            name = child.name
            text = self.clean_text(child.get_text())
            if not text:
                continue
                
            if name.startswith('h'):
                # If we have existing text, save it as a section before starting a new one
                if current_text_blocks:
                    sections.append({
                        "heading": current_heading,
                        "content": " ".join(current_text_blocks)
                    })
                    current_text_blocks = []
                current_heading = text
            else:
                current_text_blocks.append(text)
                
        # Append last section
        if current_text_blocks:
            sections.append({
                "heading": current_heading,
                "content": " ".join(current_text_blocks)
            })

        # Compile full body text
        full_text = "\n\n".join([f"## {s['heading']}\n{s['content']}" for s in sections])
        
        page_record = {
            "url": url,
            "title": title,
            "text": full_text,
            "sections": sections
        }
        self.scraped_data.append(page_record)
        
        # Extract next links to crawl
        discovered_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_link = urljoin(url, href)
            # Normalize link (remove trailing slash and lowercase domain)
            parsed_full = urlparse(full_link)
            normalized_link = f"{parsed_full.scheme}://{parsed_full.netloc.lower()}{parsed_full.path}"
            if normalized_link.endswith('/') and len(parsed_full.path) > 1:
                normalized_link = normalized_link[:-1]
                
            if self._is_valid_url(normalized_link) and normalized_link not in self.visited:
                discovered_links.append(normalized_link)
                
        return discovered_links

    def crawl(self):
        """Performs breadth-first crawl of the college website."""
        logger.info(f"Starting crawl at {self.base_url}")
        queue = [self.base_url]
        
        while queue and len(self.visited) < self.max_pages:
            current_url = queue.pop(0)
            next_links = self.scrape_page(current_url)
            for link in next_links:
                if link not in self.visited and link not in queue:
                    queue.append(link)
                    
        logger.info(f"Crawl completed. Scraped {len(self.scraped_data)} pages successfully.")
        return self.scraped_data

def scrape_and_save_json(output_path):
    """Entrypoint function to run the crawler and write to a JSON file."""
    scraper = LICTScraper(max_pages=35)
    data = scraper.crawl()
    
    # Save structured JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Structured JSON written to {output_path}")
    return len(data)

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "lict_data.json")
    scrape_and_save_json(out_path)
