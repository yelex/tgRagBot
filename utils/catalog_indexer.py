import argparse
import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
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
    old_price: Optional[float] = None
    image: Optional[str] = None
    description: Optional[str] = None
    composition: Optional[List[str]] = None
    category: Optional[str] = None
    sku: Optional[str] = None
    dimensions: Optional[Dict[str, float]] = None
    weight: Optional[float] = None
    partuids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "Название": self.name,
            "Цена": self.price,
            "Ссылка": self.link,
        }
        if self.old_price is not None:
            result["СтараяЦена"] = self.old_price
            if self.old_price > 0 and self.old_price > self.price:
                discount = int((1 - self.price / self.old_price) * 100)
                result["Скидка"] = f"{discount}%"
        if self.image:
            result["Изображение"] = self.image
        if self.description:
            result["Описание"] = self.description
        if self.composition:
            result["Состав"] = self.composition
        if self.category:
            result["Категория"] = self.category
        if self.sku:
            result["Артикул"] = self.sku
        if self.dimensions:
            result["Размеры"] = self.dimensions
        if self.weight is not None and self.weight > 0:
            result["Вес"] = self.weight
        if self.partuids:
            result["СвязанныеТовары"] = self.partuids
        return result


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

    def _extract_product_from_tilda_json(self, html: str, page_url: str) -> Optional[BouquetItem]:
        """
        Извлекает данные из встроенного JSON-блока Tilda вида:
        var product = {"uid":..., "title":"...", "text":"...", "price":"...",
                       "priceold":"...", "gallery":[...], "externalid":"...",
                       "pack_label":"lwh","pack_x":...,"pack_y":...,"pack_z":...,"pack_m":...,
                       "partuids":[...]};
        """
        # Ищем блок var product = {...}; с корректным учетом вложенных скобок
        start_idx = html.find("var product = ")
        if start_idx < 0:
            return None

        brace_start = html.find("{", start_idx)
        if brace_start < 0:
            return None

        depth = 0
        brace_end = brace_start
        for i in range(brace_start, len(html)):
            ch = html[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break

        json_str = html[brace_start:brace_end]
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        name = str(data.get("title", "")).strip()
        price_raw = str(data.get("price", "")).strip()
        price = self._parse_price(price_raw)
        if not name or price is None:
            return None

        # Старая цена
        old_price = None
        old_raw = str(data.get("priceold", "")).strip()
        if old_raw and old_raw != price_raw:
            old_price = self._parse_price(old_raw)

        # Изображение
        image = None
        gallery = data.get("gallery", [])
        if isinstance(gallery, list) and gallery:
            first = gallery[0]
            if isinstance(first, dict):
                image = first.get("img") or first.get("src") or first.get("image")
            elif isinstance(first, str):
                image = first

        # Описание (алиасы: text, description)
        description = str(data.get("text", "") or data.get("description", "") or "").strip()
        if not description:
            description = None

        # Артикул
        sku = str(data.get("externalid", "")).strip() or None

        # Размеры упаковки
        pack_label = str(data.get("pack_label", "")).strip()
        dimensions = None
        if pack_label in ("lwh", "whd", "box"):
            px = data.get("pack_x")
            py = data.get("pack_y")
            pz = data.get("pack_z")
            d: Dict[str, float] = {}
            if isinstance(px, (int, float)) and px > 0:
                d["длина"] = float(px)
            if isinstance(py, (int, float)) and py > 0:
                d["ширина"] = float(py)
            if isinstance(pz, (int, float)) and pz > 0:
                d["высота"] = float(pz)
            if d:
                dimensions = d

        # Вес (гр -> кг, если > 1000)
        weight = None
        pm = data.get("pack_m")
        if isinstance(pm, (int, float)) and pm > 0:
            w = float(pm)
            if w > 1000:
                w = round(w / 1000, 2)
            weight = w

        # Связанные товары
        partuids = None
        raw_parts = data.get("partuids")
        if isinstance(raw_parts, list) and len(raw_parts) > 0:
            partuids = [str(p) for p in raw_parts]

        # Извлекаем состав из description или из текстового блока на странице
        composition = self._extract_composition(description or "", html)

        return BouquetItem(
            name=name,
            price=price,
            link=page_url,
            old_price=old_price,
            image=image,
            description=description,
            composition=composition,
            sku=sku,
            dimensions=dimensions,
            weight=weight,
            partuids=partuids,
        )

    def _extract_composition(self, description: str, html: str) -> Optional[List[str]]:
        """Пытается извлечь список цветов в букете из описания или meta-тегов."""
        text_to_search = description
        if not text_to_search:
            # Пробуем достать из meta description
            m = re.search(
                r'<meta\s+(?:name|property)\s*=\s*["\'](?:description|og:description)["\']\s+content\s*=\s*["\']([^"\']+)["\']',
                html,
                flags=re.IGNORECASE,
            )
            if m:
                text_to_search = m.group(1)
        if not text_to_search:
            return None

        # Пробуем найти перечисление цветов через запятую после слов "с ...", "из ...", "состоит из"
        # Типичные паттерны: "с гортензией, кустовыми розами..."
        flowers = set()
        # Ищем фразу вида "с [цветок 1], [цветок 2] и [цветок N]"
        flower_patterns = re.findall(
            r'(?:с|из|состоит из|включает)\s+([а-яА-Яa-zA-Z\s,]+?)(?:\s+\.|\s+Цветы|\s+Свежие|\s+Доставка|\s+Оплата|\s+Описание|\s*$|\s*\))',
            text_to_search,
            flags=re.IGNORECASE,
        )
        for fragment in flower_patterns:
            # Разбиваем по запятым и "и"
            parts = re.split(r'[,;]|\s+и\s+', fragment)
            for part in parts:
                part = part.strip().strip(".").strip()
                if part and len(part) > 2:
                    flowers.add(part)

        # Fallback: ищем названия цветов прямо в description через известные ключевые слова
        known_flowers = [
            "роза", "розы", "эустома", "хризантем", "пион", "пионы", "маттиол", "гортензи",
            "эвкалипт", "лили", "диантус", "ромашк", "подсолнух", "гипсофил",
            "лаванд", "альстромери", "гвоздик", "ирис", "тюльпан", "орхиде",
            "ранункулюс", "гербер", "пионовидн",
        ]
        for flower in known_flowers:
            if flower in text_to_search.lower():
                # Извлекаем полное название (прилагательное + существительное)
                pattern = rf'([а-яА-Яa-zA-Z-]+\s+)?{flower}[а-яА-Яa-zA-Z-\s]*'
                for m in re.finditer(pattern, text_to_search, flags=re.IGNORECASE):
                    candidate = m.group(0).strip().strip(",").strip()
                    if candidate and len(candidate) > 2:
                        flowers.add(candidate)

        return sorted(flowers) if flowers else None

    def _extract_category_from_title(self, html: str) -> Optional[str]:
        """Извлекает категорию из <title> (например: "Сборные букеты | ...")."""
        m = re.search(r'<title>([^<]+)</title>', html, flags=re.IGNORECASE)
        if m:
            title_text = m.group(1).strip()
            # Ищем разделитель "|"
            parts = [p.strip() for p in title_text.split("|")]
            if len(parts) >= 2:
                category = parts[1].strip()
                if category and category not in ("Flori Pacco", "Цветы Москва", "Цветы"):
                    return category
        return None

    def _extract_product_from_json_ld(self, html: str, page_url: str) -> Optional[BouquetItem]:
        """Оставляет для обратной совместимости — сначала пробуем Tilda JSON."""
        # Сначала пробуем более богатый Tilda JSON
        result = self._extract_product_from_tilda_json(html, page_url)
        if result is not None:
            return result

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

                # Извлекаем image из JSON-LD
                image = None
                raw_img = entry.get("image")
                if isinstance(raw_img, str):
                    image = raw_img
                elif isinstance(raw_img, list) and raw_img:
                    image = raw_img[0] if isinstance(raw_img[0], str) else None

                # Извлекаем description
                description = str(entry.get("description", "")).strip() or None

                if name and price is not None and link:
                    item = BouquetItem(name=name, price=price, link=link, image=image, description=description)
                    # Попробуем извлечь категорию из title
                    category = self._extract_category_from_title(html)
                    if category:
                        item.category = category
                    return item
        return None

    def _extract_product_from_meta(self, html: str, page_url: str) -> Optional[BouquetItem]:
        soup = BeautifulSoup(html, "html.parser")
        item_name = soup.find(attrs={"itemprop": "name"})
        item_price = soup.find(attrs={"itemprop": "price"})
        title_meta = soup.find("meta", attrs={"property": "og:title"})
        desc_meta = soup.find("meta", attrs={"property": "og:description"})
        image_meta = soup.find("meta", attrs={"property": "og:image"})
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

        # Изображение
        image = None
        if image_meta and image_meta.get("content"):
            image = image_meta["content"].strip()

        # Описание
        description = None
        if desc_meta and desc_meta.get("content"):
            description = desc_meta["content"].strip()
        if not description:
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"].strip()

        if name and price is not None and link:
            item = BouquetItem(name=name, price=price, link=link, image=image, description=description)
            category = self._extract_category_from_title(html)
            if category:
                item.category = category
            return item
        return None

    def _fetch(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;"
                        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
                    ),
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            )
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
            html = self._fetch(link)
            if not html:
                continue

            # Сначала пытаемся Tilda JSON — он самый полный (изображение, описание, состав, размеры)
            product = self._extract_product_from_tilda_json(html, link)

            # если Tilda JSON не дал результата — пробуем JSON-LD и meta
            if product is None:
                product = self._extract_product_from_json_ld(html, link)
            if product is None:
                product = self._extract_product_from_meta(html, link)

            if product and link in found_products:
                # Если уже есть товар из списка, сохраняем более чистые имя и цену из списка
                product.name = found_products[link].name
                product.price = found_products[link].price
            if product:
                # Пробуем дозаполнить категорию, если её нет
                if not product.category:
                    product.category = self._extract_category_from_title(html)
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