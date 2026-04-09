import argparse
import json
import os
import re
import sys
import time
import unicodedata
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib import error, request


ANCHOR_PATTERN = re.compile(
    r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

TAG_PATTERN = re.compile(r"<[^>]+>")


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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


def extract_spec_link(html: str) -> Optional[str]:
    for match in ANCHOR_PATTERN.finditer(html):
        href = match.group(1).strip()
        text = TAG_PATTERN.sub(" ", match.group(2))
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        norm = normalize_text(text)
        if "xem thong so chi tiet" in norm:
            if href.startswith("http://") or href.startswith("https://"):
                return href
    return None


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
    except Exception as exc:
        return -1, json.dumps({"error": str(exc)})


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
        description="Extract and crawl 'Xem thong so chi tiet' links from existing raw files.",
    )
    parser.add_argument(
        "--input-dir",
        default="vinfast",
        help="Input subfolder in data/raw containing current raw JSON (default: vinfast)",
    )
    parser.add_argument(
        "--out-dir",
        default="vinfast_specs",
        help="Output subfolder in data/raw (default: vinfast_specs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout seconds for each API request (default: 120)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between requests (default: 0)",
    )
    parser.add_argument(
        "--unique-url",
        action="store_true",
        help="Only crawl unique spec URLs once",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    load_dotenv(project_root / ".env")

    raw_dir = base_dir / "raw"
    input_dir = raw_dir / args.input_dir
    output_dir = raw_dir / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print("Missing FIRECRAWL_API_KEY. Please set it in environment or .env file.")
        return 1

    source_files = sorted(path for path in input_dir.glob("*.json") if path.name != "manifest.json")
    if not source_files:
        print(f"No JSON files found in {input_dir}")
        return 1

    jobs: List[Dict[str, Any]] = []
    seen_urls = set()

    for source_file in source_files:
        data = json.loads(source_file.read_text(encoding="utf-8"))
        name = data.get("name", source_file.stem)
        source_url = data.get("url", "")
        html = ((data.get("firecrawl") or {}).get("data") or {}).get("html", "")
        spec_url = extract_spec_link(html)

        if not spec_url:
            jobs.append(
                {
                    "name": name,
                    "source_file": str(source_file),
                    "source_url": source_url,
                    "status": "skip_no_spec_link",
                }
            )
            continue

        if args.unique_url and spec_url in seen_urls:
            jobs.append(
                {
                    "name": name,
                    "source_file": str(source_file),
                    "source_url": source_url,
                    "spec_url": spec_url,
                    "status": "skip_duplicate_spec_url",
                }
            )
            continue

        seen_urls.add(spec_url)
        jobs.append(
            {
                "name": name,
                "source_file": str(source_file),
                "source_url": source_url,
                "spec_url": spec_url,
                "status": "pending",
            }
        )

    pending_jobs = [job for job in jobs if job["status"] == "pending"]
    if not pending_jobs:
        print("No spec links found to crawl.")
        manifest_path = output_dir / "manifest.json"
        save_json(
            manifest_path,
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "total_sources": len(source_files),
                "total_jobs": 0,
                "success": 0,
                "failed": 0,
                "items": jobs,
            },
        )
        return 0

    slug_counts: Dict[str, int] = {}
    success_count = 0

    for idx, job in enumerate(pending_jobs, start=1):
        name = job["name"]
        spec_url = job["spec_url"]

        slug = slugify(name)
        slug_counts[slug] = slug_counts.get(slug, 0) + 1
        if slug_counts[slug] > 1:
            slug = f"{slug}__{slug_counts[slug]}"

        out_file = output_dir / f"{slug}.json"

        print(f"[{idx}/{len(pending_jobs)}] Scraping spec: {name} -> {spec_url}")
        ok, result = firecrawl_scrape(spec_url, api_key=api_key, timeout=args.timeout)

        if ok:
            payload = {
                "name": name,
                "source_url": job.get("source_url"),
                "spec_url": spec_url,
                "source_file": job.get("source_file"),
                "scraped_at": int(time.time()),
                "firecrawl": result,
            }
            save_json(out_file, payload)
            job["status"] = "ok"
            job["file"] = str(out_file)
            success_count += 1
        else:
            job["status"] = "error"
            job["error"] = result

        if args.delay > 0:
            time.sleep(args.delay)

    manifest_path = output_dir / "manifest.json"
    save_json(
        manifest_path,
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "total_sources": len(source_files),
            "total_jobs": len(pending_jobs),
            "success": success_count,
            "failed": len(pending_jobs) - success_count,
            "items": jobs,
        },
    )

    print(f"Done. Success: {success_count}/{len(pending_jobs)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
