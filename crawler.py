import requests
from bs4 import BeautifulSoup
import time
import os
import urllib.parse
import json
import re # Using regex for slightly cleaner URL joining

# --- Configuration ---
START_URLS = [
    'https://rickandmorty.fandom.com/wiki/Category:Characters',
    'https://rickandmorty.fandom.com/wiki/Category:Episodes',
    'https://rickandmorty.fandom.com/wiki/Category:Locations'
]
BASE_URL = 'https://rickandmorty.fandom.com'
HEADERS = {'User-Agent': 'MyUniAssignmentCrawler/1.0 (Learning project; contact: your_email@example.com)'}
SAVE_DIR = 'fandom_image_data' # Directory to save metadata
TARGET_IMAGE_COUNT = 1500
DELAY_SECONDS = 2

# --- Global Variables ---
visited_urls = set()
image_data_list = []
urls_to_crawl = list(START_URLS) # Use a list as a queue (FIFO for BFS)

# --- Helper Functions ---

def fetch_page(url):
    """Fetches HTML content of a URL respecting politeness rules."""
    if url in visited_urls:
        # print(f"Skipping already visited: {url}")
        return None
    print(f"Fetching: {url}")
    visited_urls.add(url)

    try:
        # Wait before making the request
        print(f"  Waiting {DELAY_SECONDS} seconds...")
        time.sleep(DELAY_SECONDS)

        response = requests.get(url, headers=HEADERS, timeout=20) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Basic check for unusual content types (we want HTML)
        if 'text/html' not in response.headers.get('Content-Type', ''):
             print(f"  Skipping non-HTML content: {response.headers.get('Content-Type')}")
             return None

        return response.text

    except requests.exceptions.Timeout:
        print(f"  Error: Timeout fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during fetch: {e}")
        return None

def clean_image_url(img_src):
    """Attempts to clean Fandom image URLs to get a base version."""
    if not img_src:
        return None
    # Remove query parameters like ?cb=...
    img_src = img_src.split('?')[0]
    # Remove Fandom resizing/revision info if present
    img_src = re.sub(r'/revision/latest.*', '', img_src)
    img_src = re.sub(r'/scale-to-width-down/\d+.*', '', img_src)
    return img_src

def parse_page(html_content, page_url):
    """
    Parses HTML to find image data (if it's an article page)
    and links to follow (articles or other categories).
    Extracts context from title, alt text, caption, and FIRST FIVE valid paragraphs.
    """
    images_found = []
    links_to_follow = set() # Use a set to automatically handle duplicates within page
    soup = BeautifulSoup(html_content, 'html.parser')

    # --- 1. Find Image and Context (Primarily for Article Pages) ---
    # Target the main infobox image first, as it's usually the most relevant
    infobox = soup.find('aside', class_='portable-infobox')
    article_image_url = None
    alt_text = ""
    context_parts = [] # Initialize list to store parts of the context

    # Add Page Title to context
    page_title_tag = soup.find('h1', id='firstHeading')
    page_title = page_title_tag.get_text(strip=True) if page_title_tag else "Untitled Page"
    context_parts.append(page_title)

    # Extract Image URL, Alt Text, and Caption from Infobox
    if infobox:
        figure_tag = infobox.find('figure')
        img_tag = None
        if figure_tag:
            img_tag = figure_tag.find('img')
            caption_tag = figure_tag.find('figcaption')
            if caption_tag:
                context_parts.append(caption_tag.get_text(strip=True)) # Add caption text
        else:
             img_tag = infobox.find('img')

        if img_tag:
            img_src = img_tag.get('src') or img_tag.get('data-src')
            cleaned_url = clean_image_url(img_src)
            if cleaned_url and 'static.wikia.nocookie.net' in cleaned_url:
                 article_image_url = urllib.parse.urljoin(BASE_URL, cleaned_url)
                 alt_text = img_tag.get('alt', '')
                 if alt_text:
                     context_parts.append(alt_text) # Add alt text

    # Get text from the first FIVE valid paragraphs in the main content area
    content_body = soup.find('div', class_='mw-parser-output')
    if content_body:
        # Find all direct child paragraph tags, limit to first 5
        paragraphs = content_body.find_all('p', recursive=False)[:5]
        for p_tag in paragraphs:
            # Filter out paragraphs that are likely just containers for other things
            if not p_tag.find(['table', 'div', 'aside', 'navbox'], recursive=False):
                p_text = p_tag.get_text(strip=True)
                # Add the paragraph text if it's not empty
                if p_text:
                    context_parts.append(p_text) # Append the full paragraph text

    # If we found a primary image for the article, store it with the combined context
    if article_image_url:
        # Join all collected context parts with a space
        final_context = ' '.join(filter(None, context_parts)) # filter(None, ...) removes empty strings if any part was empty
        images_found.append({
            'image_url': article_image_url,
            'source_page': page_url,
            'alt_text': alt_text, # Keep alt_text separate if needed, but it's also in context
            'context': final_context.strip()
        })
        print(f"  Found image: {article_image_url}")


    # --- 2. Find Links to Follow (Articles and Categories) ---
    # Look in main content area and potentially category listings
    content_area = soup.find('div', id='mw-content-text') # Broader content area
    if not content_area:
         content_area = soup.body

    if content_area:
        for link_tag in content_area.find_all('a', href=True):
            href = link_tag['href']
            # Filter for relevant wiki links
            if href.startswith('/wiki/') and ':' not in href and '#' not in href:
                # ':' check avoids Special:, Category:, File:, Template: etc.
                # '#' check avoids in-page links
                full_link_url = urllib.parse.urljoin(BASE_URL, href)
                links_to_follow.add(full_link_url)
            elif href.startswith('/wiki/Category:'): # Also follow category links
                full_link_url = urllib.parse.urljoin(BASE_URL, href)
                links_to_follow.add(full_link_url)

    return images_found, list(links_to_follow) # Return links as a list

# --- Main Crawling Loop ---
def main():
    global urls_to_crawl, visited_urls, image_data_list # Allow modification

    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

    print(f"Starting crawl. Target: {TARGET_IMAGE_COUNT} images.")
    print(f"Initial queue size: {len(urls_to_crawl)}")

    while urls_to_crawl and len(image_data_list) < TARGET_IMAGE_COUNT:
        current_url = urls_to_crawl.pop(0) # Get the next URL from the front (BFS)

        if current_url in visited_urls:
            continue # Skip if already processed

        html = fetch_page(current_url)

        if html:
            try:
                images, new_links = parse_page(html, current_url)

                # Add found images if we haven't hit the limit and they are not duplicates
                for img_data in images:
                    if len(image_data_list) >= TARGET_IMAGE_COUNT:
                        break # Stop adding images if limit reached

                    # Check for duplicate image URLs before adding
                    is_duplicate = any(existing['image_url'] == img_data['image_url'] for existing in image_data_list)
                    if not is_duplicate:
                        image_data_list.append(img_data)
                        print(f"  Collected {len(image_data_list)}/{TARGET_IMAGE_COUNT} images.")

                # Add new, unvisited links to the crawl queue
                for link in new_links:
                    if link not in visited_urls and link not in urls_to_crawl:
                        urls_to_crawl.append(link)

            except Exception as e:
                print(f"  Error parsing {current_url}: {e}")
                # Optionally add error details or traceback here for debugging
                # import traceback
                # traceback.print_exc()

        print(f"  Queue size: {len(urls_to_crawl)}, Visited: {len(visited_urls)}")


    print(f"\nFinished crawling.")
    print(f"Collected data for {len(image_data_list)} images.")

    # --- Save the collected data ---
    if image_data_list:
        data_file_path = os.path.join(SAVE_DIR, 'fandom_image_metadata.json')
        try:
            with open(data_file_path, 'w', encoding='utf-8') as f:
                json.dump(image_data_list, f, indent=4, ensure_ascii=False)
            print(f"Image metadata saved to {data_file_path}")
        except IOError as e:
            print(f"Error saving data to {data_file_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during saving: {e}")
    else:
        print("No image data collected.")

if __name__ == "__main__":
    main()