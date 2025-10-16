import cv2 as cv
import numpy as np
import argparse
from math import hypot, exp

# ---------------- Tunables ----------------
SCALE = 0.5
SLOW_FACTOR = 0.5
UPDATE_SEC = 2.0            # add new dots every N seconds
FADE_SEC = 2.5              # lifetime half-life (seconds)
MAX_POINTS = 50
MIN_SEP = 16
PAIR_DIST_MAX = 150
INTENSITY_DELTA_MAX = 14
CIRCLE_R = 30
CURVE_BULGE = 0.15
LINE_THICK = 1
SHOW_LABELS = True
FONT = cv.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.7
FONT_THICK = 1
BLUR_K = 5
# ------------------------------------------

def mean_intensity_disk(gray, x, y, r=5):
    x, y, r = int(x), int(y), int(r)
    h, w = gray.shape
    x0, x1 = max(0, x - r), min(w - 1, x + r)
    y0, y1 = max(0, y - r), min(h - 1, y + r)
    if x1 < x0 or y1 < y0:
        return float(gray[min(max(0, y), h - 1), min(max(0, x), w - 1)])
    roi = gray[y0:y1 + 1, x0:x1 + 1]
    mask = np.zeros_like(roi, np.uint8)
    cv.circle(mask, (int(x - x0), int(y - y0)), r, 255, -1, cv.LINE_AA)
    return cv.mean(roi, mask=mask)[0]

def quad_curve(p0, p1, bulge=0.15, steps=32):
    p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
    mid = 0.5 * (p0 + p1)
    v = p1 - p0
    d = np.linalg.norm(v) + 1e-6
    perp = np.array([-v[1], v[0]]) / d
    c = mid + bulge * d * perp
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    pts = (1 - t)**2 * p0 + 2*(1 - t)*t*c + t**2 * p1
    return pts.astype(np.int32)

def detect_points(gray, mser, max_points):
    feats = []
    regions, _ = mser.detectRegions(gray)
    for pts in regions:
        a = cv.contourArea(pts)
        if 80 <= a <= 4000:
            M = cv.moments(pts)
            if M["m00"] > 0:
                cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                mu = mean_intensity_disk(gray, cx, cy, 5)
                feats.append((cx, cy, mu))
    if len(feats) > max_points:
        idx = np.random.choice(len(feats), max_points, replace=False)
        feats = [feats[i] for i in idx]
    return feats

def pair_indices(feats, max_dist, max_diff):
    arr = np.array(feats)
    used = set(); pairs = []
    for i,(x,y,mu) in enumerate(feats):
        if i in used: continue
        dxy = arr[:,:2] - np.array([x,y])
        dist = np.hypot(dxy[:,0], dxy[:,1])
        mask = (np.abs(arr[:,2]-mu)<=max_diff)&(dist<max_dist)
        mask[i]=False
        if not np.any(mask): continue
        j = np.argmin(np.where(mask, dist, np.inf))
        if not np.isfinite(dist[j]): continue
        pairs.append((i,j)); used|={i,j}
    return pairs

class Dot:
    def __init__(self, x,y,v):
        self.pos = np.array([x,y],np.float32)
        self.val = v
        self.age = 0.0
        self.alpha = 1.0
        self.alive = True

def main(args):
    cap = cv.VideoCapture(args.input)
    if not cap.isOpened(): raise SystemExit(f"Could not open {args.input}")
    w,h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    out_w,out_h = int(w*SCALE),int(h*SCALE)
    out_fps = max(1.0, fps*SLOW_FACTOR)
    writer=None
    if args.save:
        fourcc=cv.VideoWriter_fourcc(*"mp4v")
        writer=cv.VideoWriter(args.save,fourcc,out_fps,(out_w,out_h))
    mser=cv.MSER_create(5,80,4000)
    lk=dict(winSize=(21,21),maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,0.01))
    dots=[]
    last_gray=None; frame_idx=0; reseed=int(fps*UPDATE_SEC)
    fade_lambda=np.log(2)/FADE_SEC
    while True:
        ok,frame=cap.read()
        if not ok: break
        vis=cv.resize(frame,(out_w,out_h))
        gray=cv.cvtColor(vis,cv.COLOR_BGR2GRAY)
        if BLUR_K%2: gray=cv.GaussianBlur(gray,(BLUR_K,BLUR_K),0)
        # Optical flow update
        if last_gray is not None and dots:
            p0 = np.array([d.pos for d in dots], np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv.calcOpticalFlowPyrLK(last_gray, gray, p0, None, **lk)
            if p1 is not None and st is not None:
                p1_flat = p1.reshape(-1, 2)
                st_flat = st.reshape(-1)
                for d, n, s in zip(dots, p1_flat, st_flat):
                    if int(s) == 1:
                        d.pos = n
                else:
                    d.alive = False
            else:
            # if LK failed entirely this frame, let existing dots fade
                pass
        # Age & fade
        for d in dots:
            d.age+=1/fps
            d.alpha=float(exp(-fade_lambda*d.age))
            if d.alpha<0.03: d.alive=False
        dots=[d for d in dots if d.alive]
        # Reseed periodically
        if frame_idx%reseed==0:
            feats=detect_points(gray,mser,MAX_POINTS)
            for x,y,v in feats:
                if all(hypot(x-d.pos[0],y-d.pos[1])>MIN_SEP for d in dots):
                    dots.append(Dot(x,y,v))
        # Update brightness
        for d in dots:
            d.val=mean_intensity_disk(gray,d.pos[0],d.pos[1],5)
        # Draw
        visf=vis.astype(np.float32)/255
        overlay=np.zeros_like(visf)
        feats=[(d.pos[0],d.pos[1],d.val) for d in dots]
        pairs=pair_indices(feats,PAIR_DIST_MAX,INTENSITY_DELTA_MAX)
        for (i,j) in pairs:
            d1,d2=dots[i],dots[j]
            a=min(d1.alpha,d2.alpha)
            color=(a,a,a)
            curve=quad_curve(d1.pos,d2.pos,CURVE_BULGE)
            cv.polylines(overlay,[curve],False,color,LINE_THICK,cv.LINE_AA)
            cv.circle(overlay,(int(d1.pos[0]),int(d1.pos[1])),CIRCLE_R,color,1,cv.LINE_AA)
            cv.circle(overlay,(int(d2.pos[0]),int(d2.pos[1])),CIRCLE_R,color,1,cv.LINE_AA)
            if SHOW_LABELS:
                shared=int(round(0.5*(d1.val+d2.val)))
                mid=curve[len(curve)//2]
                cv.putText(vis,f"{shared}",(int(mid[0])+5,int(mid[1])-5),
                           FONT,FONT_SCALE,(255,255,255),FONT_THICK,cv.LINE_AA)
        vis=np.clip(visf+overlay,0,1)
        vis=(vis*255).astype(np.uint8)
        if args.display:
            cv.imshow("Cloud overlay (tracked dots)",vis)
            if (cv.waitKey(int(1000/out_fps))&0xFF)==ord('q'): break
        if writer: writer.write(vis)
        last_gray=gray.copy(); frame_idx+=1
    cap.release()
    if writer: writer.release()
    if args.display: cv.destroyAllWindows()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--save")
    ap.add_argument("--display",action="store_true")
    args=ap.parse_args()
    main(args)
