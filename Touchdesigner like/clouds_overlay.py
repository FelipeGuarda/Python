import cv2 as cv
import numpy as np
import argparse
from math import hypot

# ---------------- Tunables ----------------
SCALE = 0.7
DISPLAY_SCALE = 0.6

# Slow playback/export
SLOW_FACTOR = 1       # 0.5 = half speed
FRAME_REPEAT = 1        # repeat frames to slow even more

# Visual density
MAX_POINTS = 10        # cap total number of dots (auto-thins)
PAIR_DIST_MAX = 400
INTENSITY_DELTA_MAX = 180
MIN_SEP = 28

# Aesthetics
CIRCLE_R = 30
LINE_THICK = 1
CURVE_BULGE = 0.15
FONT = cv.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.7
FONT_THICK = 1
SHOW_LABELS = True
BLUR_K = 5
# ------------------------------------------

def mean_intensity_disk(gray, x, y, r):
    x, y, r = int(x), int(y), int(r)
    h, w = gray.shape
    if w == 0 or h == 0:
        return 0.0

    x0, x1 = max(0, x - r), min(w - 1, x + r)
    y0, y1 = max(0, y - r), min(h - 1, y + r)

    # If ROI collapses (can happen at borders), fall back to the pixel value
    if x1 < x0 or y1 < y0:
        return float(gray[min(max(0, y), h - 1), min(max(0, x), w - 1)])

    roi = gray[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return float(gray[min(max(0, y), h - 1), min(max(0, x), w - 1)])

    # Fresh, contiguous mask
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cx, cy = int(x - x0), int(y - y0)
    # Use *positional* args to avoid keyword issues on some builds
    cv.circle(mask, (cx, cy), r, 255, -1, cv.LINE_AA)

    return cv.mean(roi, mask=mask)[0]

def quad_curve(p0, p1, bulge=0.15, steps=36):
    p0 = np.array(p0, np.float32); p1 = np.array(p1, np.float32)
    mid = (p0 + p1) * 0.5
    v = p1 - p0; d = np.linalg.norm(v) + 1e-6
    perp = np.array([-v[1], v[0]]) / d
    c = mid + bulge * d * perp
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    pts = (1 - t)**2 * p0 + 2*(1 - t)*t*c + t**2 * p1
    return pts.astype(np.int32)

def draw_text_outline(img, text, org):
    cv.putText(img, text, org, FONT, FONT_SCALE, (0,0,0), FONT_THICK+2, cv.LINE_AA)
    cv.putText(img, text, org, FONT, FONT_SCALE, (255,255,255), FONT_THICK, cv.LINE_AA)

def main(args):
    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")

    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    out_w, out_h = int(w * SCALE), int(h * SCALE)
    out_fps = max(1.0, fps * SLOW_FACTOR)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    show_every_n_frames = int(fps * 1)   # every 2 seconds

    writer = None
    if args.save:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(args.save, fourcc, out_fps, (out_w, out_h))

    mser = cv.MSER_create(5, 80, 5000)

    frame_idx = 0  # before loop starts

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)
        if BLUR_K and BLUR_K % 2 == 1:
            gray = cv.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        # Only draw dots every other second
        if frame_idx % show_every_n_frames == 0:
            # --- your detection and drawing code goes here ---
            feats = []
            regions, _ = mser.detectRegions(gray)
            for pts in regions:
                cnt = pts.reshape(-1, 1, 2)
                a = cv.contourArea(cnt)
                if 80 <= a <= 4000:
                    M = cv.moments(cnt)
                    if M["m00"] > 0:
                        cx = M["m10"]/M["m00"]
                        cy = M["m01"]/M["m00"]
                        mu = mean_intensity_disk(gray, cx, cy, 5)
                        feats.append((cx, cy, mu))

        # thin and limit
        if len(feats) > MAX_POINTS:
            idx = np.random.choice(len(feats), MAX_POINTS, replace=False)
            feats = [feats[i] for i in idx]

        # Draw circles
        for x, y, _ in feats:
            cv.circle(vis, (int(x), int(y)), CIRCLE_R, (255,255,255), 1, cv.LINE_AA)
            cv.circle(vis, (int(x), int(y)), 1, (255,255,255), -1, cv.LINE_AA)

        # Pair similar brightness
        arr = np.array(feats)
        used = set()
        for i, (x, y, mu) in enumerate(feats):
            if i in used: continue
            dxy = arr[:, :2] - np.array([x, y])
            dist = np.hypot(dxy[:, 0], dxy[:, 1])
            mask = (np.abs(arr[:, 2] - mu) <= INTENSITY_DELTA_MAX) & (dist < PAIR_DIST_MAX)
            mask[i] = False
            if not np.any(mask): continue
            j = np.argmin(np.where(mask, dist, np.inf))
            if not np.isfinite(dist[j]): continue
            x2, y2, mu2 = feats[j]
            curve = quad_curve((x, y), (x2, y2), bulge=CURVE_BULGE)
            cv.polylines(vis, [curve], False, (255,255,255), LINE_THICK, cv.LINE_AA)
            if SHOW_LABELS:
                shared = int(round((mu + mu2) / 2))
                mid = curve[len(curve)//2]
                draw_text_outline(vis, f"{shared}", (int(mid[0])+5, int(mid[1])-5))
            used.add(i); used.add(j)

        # preview / write
        if args.display:
            cv.imshow("Cloud overlay", vis)
            if (cv.waitKey(int(1000 / max(1.0, fps * SLOW_FACTOR))) & 0xFF) == ord('q'):
                break
    
        if writer:
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer: writer.release()
    if args.display: cv.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--save", help="Optional output video")
    p.add_argument("--display", action="store_true")
    args = p.parse_args()
    main(args)
