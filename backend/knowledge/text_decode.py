"""Decode declared text formats without guessing or changing source newlines."""

import codecs
from typing import Literal

KnowledgeTextEncoding = Literal["unicode", "gbk"]


def decode_knowledge_text(data: bytes, encoding: KnowledgeTextEncoding = "unicode") -> str:
    """A file's BOM declares its format; unmarked text uses the selected format."""
    codec = {"unicode": "utf-8", "gbk": "gbk"}[encoding]
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        codec = "utf-32"
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        codec = "utf-16"
    elif data.startswith(codecs.BOM_UTF8):
        codec = "utf-8-sig"
    return data.decode(codec)
