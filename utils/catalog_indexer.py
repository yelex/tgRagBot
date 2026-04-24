import argparse
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger("catalog_indexer")

BASE_URL = "https://floripacco.ru"
CATALOG_URL = f"{BASE_URL}/catalog"
DEFAULT_OUTPUT = "data/bouquets.json"
SITEMAP_URLS = [
    f"{BASE_URL}/sitemap-store.xml",
    f"{BASE_URL}/sitemap.xml",
    "https://www.floripacco.ru/sitemap-store.xml",
    "https://www.floripacco.ru/sitemap.xml",
]


@dataclass
class BouquetItem:
    name: str
    price: float
    link: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "Название": self.name,
            "Цена": float(self.price),
            "Ссылка": self.link,
        }


class FloripaccoIndexer:
    def __init__(self, timeout: int = 25, max_pages: int = 80):
        self.timeout = timeout
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "ru,en;q=0.9",
            }
        )

    def _normalize_link(self, link: str) -> str:
        if not link:
            return ""
        normalized = urljoin(BASE_URL, link.strip())
        return normalized.rstrip("/")

    def _parse_price(self, raw_price: Optional[str]) -> Optional[float]:
        if not raw_price:
            return None
        digits = re.sub(r"[^\d,\.]", "", raw_price)
        if not digits:
            return None
        digits = digits.replace(",", ".")
        try:
            return float(digits)
        except ValueError:
            return None

    def _is_same_domain(self, link: str) -> bool:
        if not link:
            return False
        parsed = urlparse(link)
        if not parsed.netloc:
            return True
        return parsed.netloc.endswith("floripacco.ru")

    def _is_catalog_related(self, link: str) -> bool:
        return "/catalog" in link or "/tproduct/" in link

    def _extract_product_links(self, html: str, page_url: str) -> Set[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: Set[str] = set()

        for tag in soup.select("a[href]"):
            href = tag.get("href", "").strip()
            if not href:
                continue
            full = self._normalize_link(href)
            if self._is_same_domain(full) and self._is_catalog_related(full):
                links.add(full)

        for match in re.findall(r"https://floripacco\.ru/[^\s\"'<>]+", html):
            full = self._normalize_link(match)
            if self._is_catalog_related(full):
                links.add(full)

        logger.debug("Extracted %s links from %s", len(links), page_url)
        return links

    def _extract_products_from_list_page(self, html: str, page_url: str) -> List[BouquetItem]:
        soup = BeautifulSoup(html, "html.parser")
        products: List[BouquetItem] = []

        cards = soup.select("[data-product-gen-uid], .js-product, .t-store__card")
        for card in cards:
            link_tag = card.select_one("a[href]")
            title_tag = card.select_one(
                ".js-store-prod-name, .t-store__card__title, .t-name, .js-product-name"
            )
            price_tag = card.select_one(
                ".js-product-price, .t-store__card__price-value, .t-store__card__price, .js-store-prod-price"
            )

            if not link_tag or not title_tag:
                continue

            link = self._normalize_link(link_tag.get("href", ""))
            name = title_tag.get_text(" ", strip=True)
            price = self._parse_price(price_tag.get_text(" ", strip=True) if price_tag else None)

            if name and link and ("/tproduct/" in link or "/catalog/tproduct/" in link) and price is not None:
                products.append(BouquetItem(name=name, price=price, link=link))

        if products:
            logger.info("Found %s products directly in list page %s", len(products), page_url)
        return products

    def _extract_product_from_json_ld(self, html: str, page_url: str) -> Optional[BouquetItem]:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})

        for script in scripts:
            raw = script.string or script.get_text(strip=True)
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            candidates = payload if isinstance(payload, list) else [payload]
            for entry in candidates:
                if not isinstance(entry, dict):
                    continue
                if entry.get("@type") != "Product":
                    continue
                name = str(entry.get("name", "")).strip()
                offers = entry.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price = None
                if isinstance(offers, dict):
                    price = self._parse_price(str(offers.get("price", "")))
                link = self._normalize_link(str(entry.get("url", "") or page_url))
                if name and price is not None and link:
                    return BouquetItem(name=name, price=price, link=link)
        return None

    def _extract_product_from_meta(self, html: str, page_url: str) -> Optional[BouquetItem]:
        soup = BeautifulSoup(html, "html.parser")
        item_name = soup.find(attrs={"itemprop": "name"})
        item_price = soup.find(attrs={"itemprop": "price"})
        title_meta = soup.find("meta", attrs={"property": "og:title"})
        desc_meta = soup.find("meta", attrs={"property": "og:description"})
        canonical = soup.find("link", attrs={"rel": "canonical"})

        name = ""
        if item_name is not None:
            name = (item_name.get("content") or item_name.get_text(" ", strip=True) or "").strip()
        if not name and title_meta and title_meta.get("content"):
            name = title_meta["content"].strip()
        if not name:
            h1 = soup.find("h1")
            name = h1.get_text(" ", strip=True) if h1 else ""

        price = None
        if item_price is not None:
            price = self._parse_price(item_price.get("content") or item_price.get_text(" ", strip=True))
        if price is None and desc_meta and desc_meta.get("content"):
            price = self._parse_price(desc_meta["content"])
        if price is None:
            m = re.search(r"(\d[\d\s]{1,12})(?:\s*руб|\s*₽)", html, re.IGNORECASE)
            if m:
                price = self._parse_price(m.group(1))

        link = self._normalize_link(canonical.get("href", "") if canonical else page_url)
        if name and price is not None and link:
            return BouquetItem(name=name, price=price, link=link)
        return None

    def _fetch(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def _extract_product_links_from_sitemaps(self) -> Set[str]:
        links: Set[str] = set()
        for sitemap_url in SITEMAP_URLS:
            xml = self._fetch(sitemap_url)
            if not xml:
                continue
            for loc in re.findall(r"<loc>(.*?)</loc>", xml, flags=re.IGNORECASE):
                link = self._normalize_link(loc)
                if "/tproduct/" in link or "/catalog/tproduct/" in link:
                    links.add(link)
            if links:
                logger.info(
                    "Loaded %s product links from sitemap %s",
                    len(links),
                    sitemap_url,
                )
                return links
        return links

    def index_catalog(self) -> List[BouquetItem]:
        queue: deque[str] = deque([CATALOG_URL])
        visited: Set[str] = set()
        product_links: Set[str] = self._extract_product_links_from_sitemaps()
        found_products: Dict[str, BouquetItem] = {}

        while queue and len(visited) < self.max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            html = self._fetch(url)
            if not html:
                continue

            direct_products = self._extract_products_from_list_page(html, url)
            for product in direct_products:
                found_products[product.link] = product

            links = self._extract_product_links(html, url)
            for link in links:
                if "/tproduct/" in link:
                    product_links.add(link)
                elif link not in visited:
                    queue.append(link)

        logger.info(
            "Crawl finished: visited_pages=%s, product_links=%s, direct_products=%s",
            len(visited),
            len(product_links),
            len(found_products),
        )

        for link in sorted(product_links):
            if link in found_products:
                continue
            html = self._fetch(link)
            if not html:
                continue
            product = self._extract_product_from_json_ld(html, link)
            if product is None:
                product = self._extract_product_from_meta(html, link)
            if product:
                found_products[product.link] = product

        result = list(found_products.values())
        result.sort(key=lambda item: (item.price, item.name.lower()))
        return result


def save_bouquets(items: List[BouquetItem], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([item.to_dict() for item in items], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ручной индексатор каталога floripacco.ru -> data/bouquets.json"
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Путь до выходного bouquets.json")
    parser.add_argument("--timeout", type=int, default=25, help="Таймаут запроса в секундах")
    parser.add_argument("--max-pages", type=int, default=80, help="Максимум страниц обхода")
    parser.add_argument("--dry-run", action="store_true", help="Не записывать файл, только показать статистику")
    parser.add_argument("--verbose", action="store_true", help="Подробные логи")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    logger.info("Starting catalog indexer: output=%s", args.output)
    indexer = FloripaccoIndexer(timeout=args.timeout, max_pages=args.max_pages)
    items = indexer.index_catalog()

    if not items:
        logger.error("Не удалось собрать позиции каталога. Файл не обновлен.")
        raise SystemExit(1)

    logger.info("Collected products: %s", len(items))
    if args.dry_run:
        logger.info("Dry-run mode enabled, file was not written.")
        return

    save_bouquets(items, args.output)
    logger.info("Updated file: %s", args.output)


if __name__ == "__main__":
    main()
