"""Observe four native document imports, real preference writes, and a comparison; retry only the failed query from its actual seed."""

import asyncio

from evaluation.document_import_probe import parse_args, run_document_import_probe


async def run(args):
    return await run_document_import_probe(args, expected_seed_primary_calls=2, session_id="stage76-native")


if __name__ == "__main__":
    asyncio.run(run(parse_args(description=__doc__)))
