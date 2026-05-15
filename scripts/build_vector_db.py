from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.embeddings import EmbeddingEngine
from src.nlp.retrieval import VectorDatabase, load_dialogue_pairs


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build dialogue vector index.")
    parser.add_argument("--dialogues", default="data/raw/dialogues.txt")
    parser.add_argument("--db-path", default="data/chromadb")
    parser.add_argument("--collection", default="dialogues")
    parser.add_argument("--model", default="cointegrated/rubert-tiny2")
    parser.add_argument("--no-seed-dialogues", action="store_true")
    args = parser.parse_args()

    include_seed_dialogues = not args.no_seed_dialogues
    pairs = load_dialogue_pairs(Path(args.dialogues), include_seed_dialogues=include_seed_dialogues)
    engine = EmbeddingEngine(model_name=args.model)
    db = VectorDatabase(
        db_path=args.db_path,
        collection_name=args.collection,
        dialogues_path=args.dialogues,
        include_seed_dialogues=include_seed_dialogues,
    )
    await db.ensure_ready(engine)
    print(f"Indexed {len(pairs)} dialogue pairs into {args.db_path}/{args.collection}")


if __name__ == "__main__":
    asyncio.run(main())
