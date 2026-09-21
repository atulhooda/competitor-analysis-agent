"""The native pixel size of an image, read from its own header (PNG, JPEG, WebP).

Every cover source needs it and none of them reports it: a generated picture arrives as
bytes, and a stock photo's sized variant is not the size the API described. The site
renders the cover at its natural aspect ratio only when both numbers are known, so the
answer is ``(None, None)`` rather than a guess when the header can't be read.

No image library: the three types the site can serve each announce their size in a few
bytes near the front of the file.
"""

import struct


def dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Native pixel size read from the image's own header (PNG, JPEG, WebP)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                break
            marker, length = data[offset + 1], int.from_bytes(data[offset + 2 : offset + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
                return int(width), int(height)
            offset += 2 + length
    if data[:4] == b"RIFF" and data[8:15] == b"WEBPVP8":
        if data[12:16] == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if data[12:16] == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
    return None, None


def sniff_mime(data: bytes) -> str | None:
    """The image type read from the same headers, for a server that declares none."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


__all__ = ["dimensions", "sniff_mime"]
