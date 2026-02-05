import aiohttp
import asyncio
import json
import time
import sys

# Configuration
OUT_PATH = "beron_fast_full.jsonl"
SEARCH_URL = "https://beron.mon.bg/api/search"
DETAILS_URL = "https://beron.mon.bg/api/lexeme"
CONCURRENCY_LIMIT = 50

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://beron.mon.bg/',
    'Accept': 'application/json',
    'Accept-Language': 'bg,en;q=0.9',
}

scraped_cids = set()
total_saved = 0
start_time = time.time()


async def fetch_json(session, url, params=None, retries: int = 3, timeout: int = 10):
    for i in range(retries):
        try:
            async with session.get(url, params=params, headers=HEADERS, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    delay = 2 * (i + 1)
                    print(f"Rate limited. Waiting {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    return None
        except Exception:
            await asyncio.sleep(1)
    return None


async def crawl_prefix(session, prefix: str, queue: asyncio.Queue, fh):
    global total_saved
    data = await fetch_json(session, SEARCH_URL, params={"searchTerm": prefix})
    if not data:
        return

    if len(data) >= 5 and len(prefix) < 6:
        alphabet = "абвгдежзийклмнопрстуфхцчшщъьюя"
        for ch in alphabet:
            queue.put_nowait(prefix + ch)

    for item in data:
        cid = item.get("CID")
        phrase = item.get("PHRASE")
        if not cid or cid in scraped_cids:
            continue
        scraped_cids.add(cid)

        details = await fetch_json(session, f"{DETAILS_URL}/{cid}")
        if not details or "array" not in details:
            continue

        lemma = details.get("basicPhrase")
        pos = details.get("pos")
        forms = []
        for f_item in details["array"]:
            tags = []
            if "items" in details:
                for enr in details["items"]:
                    if enr.get("id_item") == f_item.get("id_item"):
                        tags = enr.get("props", [])
                        break
            forms.append({
                "form": f_item.get("phrase", [""])[0],
                "stress_index": f_item.get("stress", [""])[0],
                "tags": tags,
                "code": f_item.get("code", [""])[0],
            })

        entry = {"lemma": lemma, "pos": pos, "forms": forms}
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        total_saved += 1

        if total_saved % 100 == 0:
            elapsed = time.time() - start_time
            rate = total_saved / max(1e-9, elapsed)
            print(f"Saved: {total_saved} | Queue: {queue.qsize()} | Speed: {rate:.2f} w/s | Last: {phrase}")


async def worker(name: str, session, queue: asyncio.Queue, fh):
    while True:
        prefix = await queue.get()
        try:
            await crawl_prefix(session, prefix, queue, fh)
        except Exception as e:
            print(f"Worker {name} error: {e}")
        finally:
            queue.task_done()


async def run():
    queue = asyncio.Queue()
    alphabet = "абвгдежзийклмнопрстуфхцчшщъьюя"
    for ch in alphabet:
        queue.put_nowait(ch)

    print(f"Starting with {CONCURRENCY_LIMIT} concurrent connections...")
    with open(OUT_PATH, "a", encoding="utf-8") as fh:
        async with aiohttp.ClientSession() as session:
            tasks = [asyncio.create_task(worker(f"worker-{i}", session, queue, fh)) for i in range(CONCURRENCY_LIMIT)]
            await queue.join()
            for t in tasks:
                t.cancel()

    print(f"Done. Total saved: {total_saved}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())