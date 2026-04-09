import argparse
import json
import re
import unicodedata
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_tags(text: str) -> str:
    return TAG_PATTERN.sub(" ", text)


def clean_text(text: str) -> str:
    text = unescape(strip_tags(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_attribute_value(key: str, value: str) -> str:
    val = clean_text(value)
    key_fold = ascii_fold(key)

    if val in {"-", "--", "N/A", "n/a", "None", "null"}:
        return ""

    if val == "1":
        return "Có"
    if val == "0":
        # In scraped V-Car data, 0/1 is used as false/true for many flags.
        return "Không"

    if key_fold == "cam bien lui" and val in {"co", "khong"}:
        return "Có" if val == "co" else "Không"

    return val


def ascii_fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


def extract_model(name: str) -> str:
    model = re.sub(r"^VinFast\s+", "", name, flags=re.IGNORECASE).strip()
    model = re.sub(r"\s+20\d{2}$", "", model).strip()
    return model or name


def extract_url_source(html: str, fallback: str) -> str:
    match = re.search(r'<a href="(https?://[^"]+/hang-xe/[^"]+)"\s+title="VinFast">', html)
    if match:
        return match.group(1)
    return fallback


def extract_version_price(html: str) -> str:
    match = re.search(r'<span class="text-version">(.*?)</span>', html, flags=re.S)
    if not match:
        return ""
    return clean_text(match.group(1))


def extract_listed_price(version_price: str) -> str:
    # Example: "Kèm pin - 302 triệu"
    if " - " in version_price:
        return version_price.split(" - ", 1)[1].strip()
    return version_price


def extract_tag_value(html: str, label: str) -> str:
    pattern = rf'<a class="itemt"[^>]*>{re.escape(label)}\s*:\s*(.*?)</a>'
    match = re.search(pattern, html, flags=re.S | re.IGNORECASE)
    if not match:
        return ""
    return clean_text(match.group(1))


def extract_segment_from_markdown(md: str) -> str:
    def normalize_segment(value: str) -> str:
        value = clean_text(value)
        if not value:
            return ""

        for sep in [" như ", " với ", " nhưng ", " và ", " ở "]:
            if sep in value:
                value = value.split(sep, 1)[0].strip()

        value = re.sub(r"\s+", " ", value).strip(" .,-")

        token = re.search(r"\b([A-E]\+?)\b", value)
        if token:
            if len(value) <= 4:
                return token.group(1)
            if re.search(r"(cỡ|hạng|phân khúc|SUV|sedan|hatchback|crossover)", value, flags=re.IGNORECASE):
                return value
            return token.group(1)

        if re.search(r"(cỡ|hạng|siêu nhỏ|SUV|sedan|crossover|MPV|bán tải)", value, flags=re.IGNORECASE):
            if len(value.split()) <= 8:
                return value

        return ""

    text = md

    # Keep content from model heading onward, skip top navigation area.
    heading_pos = text.find("\n# [")
    if heading_pos != -1:
        text = text[heading_pos:]

    # Stop before recommendation/listing sections that add noisy segment mentions.
    for stop_marker in ["\n## Xe cùng phân khúc", "\n## So sánh", "\n## Lọc nâng cao"]:
        pos = text.find(stop_marker)
        if pos != -1:
            text = text[:pos]

    # Convert markdown links to text only.
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1", text)

    patterns = [
        r"(?:nằm\s+ở|thuộc|tham\s+gia\s+vào|chung)\s+phân\s+khúc\s+([^,\.\n]+)",
        r"phân\s+khúc\s+([^,\.\n]+)",
    ]

    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE)
        if match:
            candidate = normalize_segment(match.group(1))
            if candidate and "http" not in candidate and "vnexpress" not in candidate:
                return candidate

    # Fallback on folded text for badly encoded strings.
    folded = ascii_fold(text)
    for pat in [
        r"(?:nam\s+o|thuoc|tham\s+gia\s+vao|chung)\s+phan\s+khuc\s+([^,\.\n]+)",
        r"phan\s+khuc\s+([^,\.\n]+)",
    ]:
        match2 = re.search(pat, folded, flags=re.IGNORECASE)
        if match2:
            candidate = normalize_segment(match2.group(1).strip())
            if candidate and "http" not in candidate and "vnexpress" not in candidate:
                return candidate

    return ""


def extract_version_id_from_url(spec_url: str) -> Optional[str]:
    match = re.search(r"-(\d+)$", spec_url.strip())
    if match:
        return match.group(1)
    return None


def extract_version_blocks(html: str) -> List[Tuple[str, str]]:
    # Returns list of (version_id, html_slice_for_that_version)
    starts = list(
        re.finditer(
            r'<div class="list-collaps[^\"]*list-version-infor[^\"]*"[^>]*data-version-id="(\d+)"[^>]*>',
            html,
            flags=re.S,
        )
    )
    if not starts:
        return []

    blocks: List[Tuple[str, str]] = []
    right_bar_idx = html.find('<div class="slidebar-right"')

    for idx, match in enumerate(starts):
        version_id = match.group(1)
        start_pos = match.start()

        if idx + 1 < len(starts):
            end_pos = starts[idx + 1].start()
        elif right_bar_idx != -1 and right_bar_idx > start_pos:
            end_pos = right_bar_idx
        else:
            end_pos = len(html)

        blocks.append((version_id, html[start_pos:end_pos]))

    return blocks


def parse_specs_from_html(html: str, spec_url: str = "") -> List[Dict[str, Any]]:
    target_version_id = extract_version_id_from_url(spec_url) if spec_url else None
    version_blocks = extract_version_blocks(html)

    selected_blocks: List[str] = []
    if version_blocks:
        if target_version_id:
            selected_blocks = [block for version_id, block in version_blocks if version_id == target_version_id]
        if not selected_blocks:
            selected_blocks = [version_blocks[0][1]]
    else:
        selected_blocks = [html]

    specs: List[Dict[str, Any]] = []
    seen_categories = set()

    for body in selected_blocks:
        category_blocks = re.findall(
            r'<li class="collaps[^\"]*">\s*<div class="collapsed">(.*?)</div>\s*<div class="collapse"[^>]*>\s*<ul>(.*?)</ul>\s*</div>\s*</li>',
            body,
            flags=re.S,
        )

        for category_raw, rows_html in category_blocks:
            category = clean_text(category_raw)
            attributes: Dict[str, str] = {}

            rows = re.findall(
                r'<li>\s*<div class="td1">\s*<b>(.*?)</b>\s*</div>\s*<div class="td2">(.*?)</div>\s*</li>',
                rows_html,
                flags=re.S,
            )

            for key_raw, value_raw in rows:
                key = clean_text(key_raw)
                value: str
                if "#check" in value_raw:
                    value = "Có"
                elif "#cancel" in value_raw:
                    value = "Không"
                else:
                    value = normalize_attribute_value(key, value_raw)

                attributes[key] = value

            if category and attributes and category not in seen_categories:
                specs.append({"category": category, "attributes": attributes})
                seen_categories.add(category)

    return specs


def build_clean_record(spec_doc: Dict[str, Any]) -> Dict[str, Any]:
    name = spec_doc.get("name", "")
    source_url = spec_doc.get("source_url", "")
    spec_url = spec_doc.get("spec_url", "")
    source_file = spec_doc.get("source_file", "")

    firecrawl_data = (spec_doc.get("firecrawl") or {}).get("data") or {}
    html = firecrawl_data.get("html", "")

    source_md = ""
    if source_file:
        source_path = Path(source_file)
        if source_path.exists():
            try:
                source_doc = json.loads(source_path.read_text(encoding="utf-8"))
                source_md = ((source_doc.get("firecrawl") or {}).get("data") or {}).get("markdown", "")
            except Exception:
                source_md = ""

    version_price = extract_version_price(html)
    listed_price = extract_listed_price(version_price)
    car_type = extract_tag_value(html, "Loại xe")
    origin = extract_tag_value(html, "Xuất xứ")
    segment = extract_segment_from_markdown(source_md)

    return {
        "car_model": extract_model(name),
        "url_source": extract_url_source(html, source_url),
        "price": {
            "version_display": version_price,
            "listed_price": listed_price,
        },
        "vehicle_type": car_type,
        "origin": origin,
        "segment": segment,
        "specifications": parse_specs_from_html(html, spec_url=spec_url),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cleaned VinFast spec JSON from crawled raw data.")
    parser.add_argument("--input-dir", default="vinfast_specs", help="Input subfolder in data/raw")
    parser.add_argument("--out-dir", default="vinfast_clean", help="Output subfolder in data/raw")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "raw"
    input_dir = raw_dir / args.input_dir
    output_dir = raw_dir / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    source_files = sorted(path for path in input_dir.glob("*.json") if path.name != "manifest.json")
    if not source_files:
        print(f"No source files in: {input_dir}")
        return 1

    all_records: List[Dict[str, Any]] = []
    manifest_items: List[Dict[str, Any]] = []

    for source_path in source_files:
        source_doc = json.loads(source_path.read_text(encoding="utf-8"))
        record = build_clean_record(source_doc)

        out_name = source_path.name
        out_path = output_dir / out_name

        payload = [record]
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        all_records.append(record)
        manifest_items.append(
            {
                "source_file": str(source_path),
                "output_file": str(out_path),
                "car_model": record.get("car_model", ""),
                "spec_categories": len(record.get("specifications", [])),
            }
        )

    all_path = output_dir / "all_cars.json"
    all_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_files": len(source_files),
        "all_cars_file": str(all_path),
        "items": manifest_items,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Clean files: {len(source_files)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
