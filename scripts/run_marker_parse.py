#!/usr/bin/env python3
import asyncio
import traceback
import sys
import time
from pathlib import Path


async def main():
    try:
        print("about to import MarkerPDFParser", flush=True)
        from pdfkb.parsers.parser_marker import MarkerPDFParser

        print("imported MarkerPDFParser", flush=True)
        cfg = {"disable_ocr": True, "disable_table_rec": True, "debug": True}
        parser = MarkerPDFParser(config=cfg, cache_dir=Path("/tmp"))
        print("created parser, starting parse with 600s timeout", flush=True)
        start = time.time()
        try:
            result = await asyncio.wait_for(parser.parse(Path("/app/documents/sample.pdf")), timeout=600)
            print("parse finished in", time.time() - start, "seconds", flush=True)
            print("Metadata keys:", list(result.metadata.keys()), flush=True)
            if result.pages:
                text = result.pages[0].markdown_content
                print("\nFirst page excerpt (first 800 chars):", flush=True)
                print(text[:800], flush=True)
        except asyncio.TimeoutError:
            print("Parse timed out after 600s (likely downloading models or heavy processing).", flush=True)
    except Exception as e:
        print("Parse failed:", e, file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
