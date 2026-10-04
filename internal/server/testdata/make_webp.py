"""Generate the WebP fixtures for internal/server (decodeImage test).

quadrants.webp (lossless, VP8L) and quadrants-lossy.webp (VP8): a 24x16 image whose four 12x8
quadrants are red, green, blue and white. The test checks the lossless pixels exactly and the
lossy ones within a tolerance.
"""
import sys
from PIL import Image

out = sys.argv[1]
im = Image.new("RGB", (24, 16))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
for qi, c in enumerate(colors):
    x0, y0 = (qi % 2) * 12, (qi // 2) * 8
    for y in range(y0, y0 + 8):
        for x in range(x0, x0 + 12):
            im.putpixel((x, y), c)
im.save(out + "/quadrants.webp", "WEBP", lossless=True)
im.save(out + "/quadrants-lossy.webp", "WEBP", quality=90)
