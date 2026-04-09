import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request


HEADING_LINK_PATTERN = re.compile(
    r"^\s{0,3}#{1,6}\s+\[([^\]]+)\]\((https?://[^\s\)]+)(?:\s+['\"][^'\"]*['\"])?\)",
    re.IGNORECASE,
)


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "unknown"


def parse_heading_links(md_path: Path, dedupe_url: bool = False) -> List[Dict[str, str]]:
    content = md_path.read_text(encoding="utf-8")
    vehicles: List[Dict[str, str]] = []
    seen_urls = set()

    for line in content.splitlines():
        match = HEADING_LINK_PATTERN.match(line)
        if not match:
            continue

        name = match.group(1).strip()
        url = match.group(2).strip()

        if dedupe_url:
            if url in seen_urls:
                continue
            seen_urls.add(url)

        vehicles.append({"name": name, "url": url})

    return vehicles


def _post_json(url: str, payload: Dict, api_key: str, timeout: int) -> Tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return exc.code, response_body


def firecrawl_scrape(target_url: str, api_key: str, timeout: int = 120) -> Tuple[bool, Dict]:
    payload = {
        "url": target_url,
        "formats": ["markdown", "html"],
        "onlyMainContent": True,
    }

    endpoints = [
        "https://api.firecrawl.dev/v1/scrape",
        "https://api.firecrawl.dev/v0/scrape",
    ]

    last_error: Optional[Dict] = None
    for endpoint in endpoints:
        status, text = _post_json(endpoint, payload, api_key=api_key, timeout=timeout)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw": text}

        if 200 <= status < 300:
            return True, data

        last_error = {
            "endpoint": endpoint,
            "status": status,
            "response": data,
        }

    return False, last_error or {"error": "unknown error"}


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract vehicle links from total.md headings and scrape with Firecrawl.")
    parser.add_argument(
        "--input",
        default="total.md",
        help="Input markdown file name in data/raw (default: total.md)",
    )
    parser.add_argument(
        "--out-dir",
        default="vinfast",
        help="Output subfolder name inside data/raw (default: vinfast)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between requests (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout seconds for each API request (default: 120)",
    )
    parser.add_argument(
        "--dedupe-url",
        action="store_true",
        help="Only keep unique URLs (default: keep all heading links)",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    load_dotenv(project_root / ".env")

    raw_dir = base_dir / "raw"
    input_path = raw_dir / args.input
    output_dir = raw_dir / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("Missing FIRECRAWL_API_KEY. Please set it in environment or .env file.")
        return 1

    vehicles = parse_heading_links(input_path, dedupe_url=args.dedupe_url)
    if not vehicles:
        print(f"No heading links found in {input_path}")
        return 1

    manifest: List[Dict] = []
    success_count = 0

    slug_counts: Dict[str, int] = {}
    for index, vehicle in enumerate(vehicles, start=1):
        name = vehicle["name"]
        url = vehicle["url"]
        slug = slugify(name)

        slug_counts[slug] = slug_counts.get(slug, 0) + 1
        if slug_counts[slug] > 1:
            slug = f"{slug}__{slug_counts[slug]}"

        file_path = output_dir / f"{slug}.json"

        print(f"[{index}/{len(vehicles)}] Scraping: {name} -> {url}")
        ok, result = firecrawl_scrape(url, api_key=api_key, timeout=args.timeout)

        if ok:
            payload = {
                "name": name,
                "url": url,
                "scraped_at": int(time.time()),
                "firecrawl": result,
            }
            save_json(file_path, payload)
            manifest.append(
                {
                    "name": name,
                    "url": url,
                    "status": "ok",
                    "file": str(file_path),
                }
            )
            success_count += 1
        else:
            manifest.append(
                {
                    "name": name,
                    "url": url,
                    "status": "error",
                    "error": result,
                }
            )

        if args.delay > 0:
            time.sleep(args.delay)

    manifest_path = output_dir / "manifest.json"
    save_json(
        manifest_path,
        {
            "input": str(input_path),
            "total": len(vehicles),
            "success": success_count,
            "failed": len(vehicles) - success_count,
            "items": manifest,
        },
    )

    print(f"Done. Success: {success_count}/{len(vehicles)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
