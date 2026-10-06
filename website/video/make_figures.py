import json, textwrap
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/home/trung/trung_workdir/vision_serve_video/"
F = lambda n, s: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{n}.ttf", s)
INK, MUTED, ACC, BAD, GOOD = (30, 36, 46), (110, 120, 135), (42, 120, 214), (206, 70, 50), (40, 150, 120)

# 1. a wrong setting is caught -----------------------------------------------------------------------------------
W, H = 1600, 640
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
def paste(path, x, y, s, label, col):
    p = Image.open(path).convert("RGB").resize((s, s)); im.paste(p, (x, y + 44)); d.rectangle([x, y + 44, x + s, y + 44 + s], outline=col, width=5)
    d.text((x, y), label, font=F("DejaVuSans-Bold", 28), fill=col)
paste(R + "out/ok-input.png", 20, 20, 520, "Right: photo stretched to fit", GOOD)
paste(R + "out/wrongprep-input.png", 570, 20, 520, "Wrong: black bars added", BAD)
log = open(R + "logs/07-check-wrong.log", errors="replace").read().splitlines()
fail = next(l for l in log if l.startswith("FAIL:"))
x0 = 1120
d.text((x0, 20), "What the check reports", font=F("DejaVuSans-Bold", 28), fill=BAD)
y = 74
for block in (fail, "Preprocessing: difference 67.4 gray levels (out of 255) over 8 photos, where more than 8 costs accuracy.",
              "Fix: set `input.letterbox: false` in manifest.yaml"):
    for ln in textwrap.wrap(block, 34):
        d.text((x0, y), ln, font=F("DejaVuSans", 22), fill=INK); y += 30
    y += 16
im.save("figures/check_wrong.png")

# 2. how many layers tolerate each format ---------------------------------------------------------------------------
S = json.load(open(R + "out/sensitivity.json"))["layers"]
fm = [("int8", "8-bit"), ("fp16", "16-bit"), ("int4", "4-bit")]
bins = [("safe  (error below 0.01)", lambda e: e < 0.01, (60, 170, 150)), ("mildly sensitive", lambda e: 0.01 <= e <= 0.05, (240, 160, 40)), ("fragile  (error above 0.05)", lambda e: e > 0.05, BAD)]
fig, ax = plt.subplots(figsize=(10, 4.9), dpi=160)
bottom = [0, 0, 0]
for name, pred, col in bins:
    v = [sum(1 for l in S if f in l["errs"] and pred(l["errs"][f])) for f, _ in fm]
    ax.bar(range(3), v, 0.55, bottom=bottom, color=[c / 255 for c in col], label=name)
    for i, (b, vv) in enumerate(zip(bottom, v)):
        if vv >= 12: ax.text(i, b + vv / 2, str(vv), ha="center", va="center", color="white", fontsize=20, fontweight="bold")
        elif vv: ax.text(i + 0.33, b + vv / 2, str(vv), ha="left", va="center", color=[c / 255 for c in col], fontsize=20, fontweight="bold")
    bottom = [a + b for a, b in zip(bottom, v)]
tot = [sum(1 for l in S if f in l["errs"]) for f, _ in fm]
ax.set_xticks(range(3)); ax.set_xticklabels([f"{n} numbers\n({t} layers)" for (f, n), t in zip(fm, tot)], fontsize=15)
ax.set_ylabel("number of layers", fontsize=14); ax.legend(fontsize=13, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figures/sens.png"); plt.close(fig)

# 3. the same photo at three precisions ----------------------------------------------------------------------------
W, H = 1600, 640
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
cols = [("Original, 32-bit", "out/fp32", ACC), ("8-bit everywhere", "out/var_int8", BAD), ("4-bit everywhere", "out/var_int4", (150, 90, 200))]
for i, (lab, dirn, col) in enumerate(cols):
    x = 20 + i * 525
    p = Image.open(R + dirn + "/000000039769.png").convert("RGB"); p.thumbnail((500, 380))
    d.text((x, 14), lab, font=F("DejaVuSans-Bold", 28), fill=col)
    im.paste(p, (x, 58)); d.rectangle([x, 58, x + p.width, 58 + p.height], outline=col, width=4)
notes = ["16-bit and mixed precision: same four detections, confidence changed by 0.001 or less.",
         "8-bit everywhere: two false detections appear (cat 0.65, bed 0.54).",
         "4-bit everywhere: the same four objects, but every confidence is lower, by 0.10 on average."]
y = 470
for t in notes:
    d.text((20, y), t, font=F("DejaVuSans", 25), fill=INK); y += 40
im.save("figures/photos.png")
print("figures ok")
