import cv2 as cv
import numpy as np
import argparse
from math import hypot

# ----------------- Tunables -----------------
SCALE = 0.75                 # downscale for speed
HISTORY = 400                # BG subtractor history
VAR_THR = 16                 # sensitivity
DETECT_SHADOWS = True

AREA_MIN = 800               # min area for animal contour (after scaling)
AREA_MAX = 300_000           # max area for animal contour

SAMPLES = 80                 # how many dots around the contour
NEIGH_RADIUS = 7             # radius (px) to average brightness at a dot
PAIR_DIST_MAX = 120          # max geometric distance to pair
INTENSITY_DELTA_MAX = 12     # max |I1-I2| (0–255) to pair
MIN_SEP = 28                 # min distance between dots after sampling (post-thinning)

CURVE_BULGE = 0.20           # 0=straight; 0.1–0.3 is nice
LINE_THICK = 1
DOT_R = 40                    # dot radius
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
BLUR_K = 3                   # small denoise on gray
DISPLAY_SCALE = 0.5  # affects only the preview window

# --------------------------------------------

def draw_text(img, text, org, color=(255,255,255)):
    cv.putText(img, text, org, FONT, FONT_SCALE, (0,0,0), FONT_THICK+2, cv.LINE_AA)
    cv.putText(img, text, org, FONT, FONT_SCALE, color, FONT_THICK, cv.LINE_AA)

def quad_curve(p0, p1, bulge=0.18, steps=36):
    p0 = np.array(p0, np.float32); p1 = np.array(p1, np.float32)
    mid = 0.5*(p0+p1)
    v = p1 - p0
    d = np.linalg.norm(v)+1e-6
    perp = np.array([-v[1], v[0]])/d
    c = mid + bulge*d*perp
    t = np.linspace(0,1,steps).reshape(-1,1)
    pts = (1-t)**2*p0 + 2*(1-t)*t*c + (t**2)*p1
    return pts.astype(np.int32)

def mask_mean_intensity(gray, pt, r):
    x,y = int(pt[0]), int(pt[1])
    h,w = gray.shape
    x0,x1 = max(0,x-r), min(w-1,x+r)
    y0,y1 = max(0,y-r), min(h-1,y+r)
    roi = gray[y0:y1+1, x0:x1+1]
    mask = np.zeros_like(roi, dtype=np.uint8)
    cv.circle(mask, (x-x0, y-y0), r, 255, -1)
    m = cv.mean(roi, mask=mask)[0]
    return m

def resample_contour(cnt, n_points):
    """Return n_points roughly equally spaced along the closed contour."""
    pts = cnt.reshape(-1,2).astype(np.float32)
    # ensure closed
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    # cumulative arc length
    deltas = np.diff(pts, axis=0)
    seglen = np.hypot(deltas[:,0], deltas[:,1])
    L = np.concatenate([[0], np.cumsum(seglen)])
    total = L[-1]
    if total < 1e-3:
        return np.repeat(pts[:1], n_points, axis=0)

    targets = np.linspace(0, total, n_points+1)[:-1]  # drop last (same as first)
    res = []
    j = 0
    for t in targets:
        while j+1 < len(L) and L[j+1] < t:
            j += 1
        # interpolate between pts[j] and pts[j+1]
        if j+1 >= len(pts):
            res.append(pts[-1])
            continue
        seg_start = L[j]
        seg_len = max(1e-6, L[j+1] - L[j])
        alpha = (t - seg_start)/seg_len
        p = (1-alpha)*pts[j] + alpha*pts[j+1]
        res.append(p)
    return np.array(res, dtype=np.float32)

def thin_points(points, min_sep):
    """Spatial thinning on (x,y,val)."""
    kept = []
    cell = max(4, int(min_sep//2))
    grid = {}
    def key(x,y): return (int(x)//cell, int(y)//cell)
    for x,y,v in points:
        k = key(x,y)
        ok = True
        for nx in (k[0]-1,k[0],k[0]+1):
            for ny in (k[1]-1,k[1],k[1]+1):
                for idx in grid.get((nx,ny), []):
                    x2,y2,_ = kept[idx]
                    if hypot(x-x2, y-y2) < min_sep:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok:
            grid.setdefault(k, []).append(len(kept))
            kept.append((x,y,v))
    return kept

def main(args):
    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open: {args.input}")

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    out_w, out_h = int(w*SCALE), int(h*SCALE)

    writer = None
    if args.save:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(args.save, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise SystemExit(f"Could not create output: {args.save}")

    bg = cv.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THR, detectShadows=DETECT_SHADOWS)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv.resize(frame, (out_w, out_h), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if BLUR_K and BLUR_K % 2 == 1:
            gray = cv.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

        fg = bg.apply(gray)
        if DETECT_SHADOWS:
            _, fg = cv.threshold(fg, 200, 255, cv.THRESH_BINARY)  # drop shadows

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel, 1)
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel, 2)

        cnts, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        target = None
        if cnts:
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)
            for c in cnts:
                a = cv.contourArea(c)
                if AREA_MIN <= a <= AREA_MAX:
                    target = c
                    break

        vis = frame.copy()

        if target is not None:
            # Sample points around the contour
            samples = resample_contour(target, SAMPLES)
            # Compute local brightness for each sample
            feats = []
            for p in samples:
                mu = mask_mean_intensity(gray, p, NEIGH_RADIUS)
                feats.append((float(p[0]), float(p[1]), mu))

            # Thin samples to keep it clean
            feats = thin_points(feats, MIN_SEP)

            # Draw dots
            for x,y,mu in feats:
                cv.circle(vis, (int(x),int(y)), DOT_R, (255,255,255), -1, cv.LINE_AA)

            # Pair by proximity + similar brightness (greedy, non-overlapping)
            feats_arr = np.array(feats) if feats else np.empty((0,3))
            used = set()
            for i, (x,y,mu) in enumerate(feats):
                if i in used: continue
                if len(feats) < 2: break
                dxy = feats_arr[:,:2] - np.array([x,y])
                dist = np.hypot(dxy[:,0], dxy[:,1])
                idx = np.arange(len(feats))
                mask = (idx != i)
                if used:
                    mask &= ~np.isin(idx, list(used))
                mask &= (dist < PAIR_DIST_MAX)
                mask &= (np.abs(feats_arr[:,2]-mu) <= INTENSITY_DELTA_MAX)

                if not np.any(mask): continue
                j = np.argmin(np.where(mask, dist, np.inf))
                if not np.isfinite(dist[j]): continue

                x2,y2,mu2 = feats[j]
                curve = quad_curve((x,y), (x2,y2), bulge=CURVE_BULGE, steps=36)
                cv.polylines(vis, [curve], False, (255,255,255), LINE_THICK, cv.LINE_AA)

                # Label with shared brightness (mean of the two)
                shared = int(round(0.5*(mu+mu2)))
                mid = curve[curve.shape[0]//2]
                draw_text(vis, f"{shared}", (int(mid[0])+6, int(mid[1])-6))
                used.add(i); used.add(j)

        # HUD
        draw_text(vis, "Contour pairs: dots = samples, curves = similar brightness", (10, 24))

        if args.display:
            cv.namedWindow("Contour-based pairing overlay", cv.WINDOW_NORMAL)
            if DISPLAY_SCALE != 1.0:
                pv = cv.resize(vis, (int(vis.shape[1]*DISPLAY_SCALE), int(vis.shape[0]*DISPLAY_SCALE)))
            else:
                pv = vis
            cv.imshow("Contour-based pairing overlay", pv)
            if (cv.waitKey(1) & 0xFF) == ord('q'):
                break
        if writer:
            writer.write(vis)

    cap.release()
    if writer: writer.release()
    if args.display: cv.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input video (.mov/.mp4)")
    ap.add_argument("--save", help="Optional output video (.mp4)")
    ap.add_argument("--display", action="store_true", help="Show preview window")
    args = ap.parse_args()
    main(args)
