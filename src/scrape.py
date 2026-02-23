import requests
import re
from bs4 import BeautifulSoup
import csv
import time
import random
import logging
from typing import List, Dict
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = "https://www.patpat.lk/en/sri-lanka/vehicle/car"
MAX_PAGES = 100

# Updated to match your repo structure (data/raw/)
OUTPUT_FILE = Path("data/raw/vehicles_raw.csv")

# Delay settings (ethical scraping)
MIN_DELAY = 2.0
MAX_DELAY = 5.0
REQUEST_TIMEOUT = 30

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_soup(url: str) -> BeautifulSoup | None:
    """Fetch a page and return a BeautifulSoup object, or None on error."""
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))  # polite delay
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.content, "lxml")
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


def parse_listing_cards(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    Extract basic information from each vehicle listing on a results page.
    """
    records: List[Dict[str, str]] = []
    result_item_selector = "a[href*='/ad/vehicle/']"
    anchors = soup.select(result_item_selector)
    
    if not anchors:
        logger.warning("No listing anchors found. The site layout may have changed.")
        return records

    year_re = re.compile(r"\b(19|20)\d{2}\b")
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}")

    for a in anchors:
        text = a.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        if not text:
            continue

        parts = text.split()
        year_idx = None
        for idx, token in enumerate(parts):
            if year_re.fullmatch(token):
                year_idx = idx
                break
        if year_idx is None:
            continue

        title = " ".join(parts[: year_idx + 1])
        price = None
        metadata = None
        mileage = None
        i = year_idx + 1
        
        if i < len(parts) and (parts[i].lower().startswith("rs") or parts[i].lower() == "negotiable"):
            price_tokens = [parts[i]]
            i += 1
            while i < len(parts) and not date_re.match(parts[i]):
                price_tokens.append(parts[i])
                i += 1
            price = " ".join(price_tokens)

        if i < len(parts) and date_re.match(parts[i]):
            date_str = parts[i]
            i += 1
            if i < len(parts) and parts[i] == "|":
                i += 1
            loc_tokens = []
            while i < len(parts) and not parts[i].lower().endswith("km"):
                loc_tokens.append(parts[i])
                i += 1
            location = " ".join(loc_tokens)
            metadata = f"{date_str} | {location}" if location else date_str
            if i + 1 <= len(parts):
                mileage_tokens = parts[i : i + 2]
                mileage = " ".join(mileage_tokens).strip()

        href = a.get("href", "")
        full_url = urljoin(BASE_URL, href) if href and href.startswith("/") else href

        record = {
            "Raw_Title": title or None,
            "Raw_Price": price or None,
            "Raw_Mileage": mileage or None,
            "Raw_Metadata": metadata or None,
            "Listing_URL": full_url,
        }
        
        if any(record.values()):
            records.append(record)

    return records


DETAIL_LABELS = [
    "Mileage", "Engine/Motor Capacity", "Transmission", "Manufacturer",
    "Model Year", "Condition", "Model", "Fuel Type", "Colour",
    "Vehicle Type", "Power", "Register Year",
]


def get_vehicle_details(url: str) -> Dict[str, str]:
    """Fetch a vehicle detail page and extract common attributes."""
    details: Dict[str, str] = {}
    soup = get_soup(url)
    if soup is None:
        return details

    full_text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    
    for i, line in enumerate(lines[:-1]):
        for label in DETAIL_LABELS:
            if line.lower() == label.lower():
                value = lines[i + 1]
                details[label] = value
                break

    return details


def scrape_patpat(max_pages: int = MAX_PAGES, fetch_details: bool = True) -> List[Dict[str, str]]:
    """
    Paginate through patpat.lk and collect listing data.
    
    Args:
        max_pages: Number of search result pages to scrape.
        fetch_details: Whether to follow each listing URL and extract detailed attributes.
                      Set to False for faster scraping (basic fields only).
    """
    all_records: List[Dict[str, str]] = []

    for page in range(1, max_pages + 1):
        if page == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}?page={page}"

        logger.info(f"Scraping page {page}/{max_pages}: {url}")
        soup = get_soup(url)
        if soup is None:
            logger.warning(f"Skipping page {page} due to fetch error.")
            continue

        page_records = parse_listing_cards(soup)
        logger.info(f"Found {len(page_records)} records on page {page}")

        for record in page_records:
            price_str = (record.get("Raw_Price") or "").strip().lower()
            if not price_str or price_str.startswith("negotiable"):
                continue

            if fetch_details:
                listing_url = record.get("Listing_URL")
                if listing_url:
                    details = get_vehicle_details(listing_url)
                    record.update(details)

            record.pop("Listing_URL", None)
            all_records.append(record)

    logger.info(f"Total records scraped (after filtering): {len(all_records)}")
    return all_records


def save_to_csv(records: List[Dict[str, str]], filepath: Path = OUTPUT_FILE) -> None:
    """Save a list of dictionaries to a CSV file."""
    if not records:
        logger.warning("No records to save. CSV will not be created.")
        return

    # Ensure output directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())
    all_keys.discard("Listing_URL")
    
    core_fields = ["Raw_Title", "Raw_Price", "Raw_Mileage", "Raw_Metadata"]
    other_fields = sorted(k for k in all_keys if k not in core_fields)
    fieldnames = core_fields + other_fields

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved {len(records)} records to {filepath}")


if __name__ == "__main__":
    logger.info("Starting patpat.lk scraping...")
    logger.info(f"Output will be saved to: {OUTPUT_FILE}")
    
    # Set fetch_details=False for quick testing (5x faster)
    data = scrape_patpat(MAX_PAGES, fetch_details=True)
    save_to_csv(data, OUTPUT_FILE)
    logger.info("Scraping finished.")