"""Compare feature configurations on the two test pages.

Metrics
- MRR / hits@1 / hits@3 : for each known correct word pair (a, b), the rank
  of b among all B tokens for query a, and of a among all A tokens for
  query b (token-level scores, threshold-free).
- matches at the default acceptance rule, split into known-correct,
  known-wrong and unjudged (unjudged pairs are rendered for inspection).
"""
import csv, dataclasses, pickle, sys
import numpy as np, cv2
import cross_spotting as cs, word_spotting as ws

GOOD = {('c1_l26_t1','c1_l22_t1'),('c1_l2_t1','c2_l2_t1'),('c1_l17_t2','c1_l21_t7'),
        ('c1_l9_t6','c1_l6_t7'),('c1_l12_t4','c1_l21_t2'),('c1_l30_t3','c2_l7_t4'),
        ('c1_l26_t1','c1_l6_t2'),('c1_l9_t6','c1_l16_t5'),('c1_l3_t3','c2_l3_t3'),
        ('c2_l6_t1','c2_l7_t4'),('c1_l14_t4','c1_l12_t3')}
JUDGED = {(r['000126_candidate'].split('[')[0], r['mstest_candidate'].split('[')[0])
          for r in csv.DictReader(open('cross_ctx/cross_matches.csv'))}
WRONG = JUDGED - GOOD

CONFIGS = {
    "v4 baseline (HOG + profile DTW)":        dict(whiten=False, hog_weight=1, dtw_weight=1, hog_dtw_weight=0),
    "whitened HOG + profile DTW":             dict(whiten=True,  hog_weight=1, dtw_weight=1, hog_dtw_weight=0),
    "HOG + profile DTW + HOG-DTW":            dict(whiten=False, hog_weight=1, dtw_weight=1, hog_dtw_weight=1),
    "whitened HOG + profile DTW + HOG-DTW":   dict(whiten=True,  hog_weight=1, dtw_weight=1, hog_dtw_weight=1),
    "whitened HOG + HOG-DTW":                 dict(whiten=True,  hog_weight=1, dtw_weight=0, hog_dtw_weight=1),
}

def ranks(T, a_ids, b_ids):
    out = []
    for pa, pb in GOOD:
        p, q = a_ids.index(pa), b_ids.index(pb)
        out.append(1 + int((T[p] > T[p, q]).sum()))        # A -> B
        out.append(1 + int((T[:, q] > T[p, q]).sum()))     # B -> A
    return np.array(out)

def main():
    blob = open('pages_eval.pkl', 'rb').read()
    for name, kw in CONFIGS.items():
        A, B = pickle.loads(blob)
        cp = dataclasses.replace(cs.CrossParams(), **kw)
        cs.prepare_features(A, B, cp, ws.SpotParams())
        P = cs.pair_scores(A, B, cp)
        T, _, _ = cs.token_scores(A, B, P["score"], cp)
        a_ids = list(dict.fromkeys(c.parent for c in A.cands))
        b_ids = list(dict.fromkeys(c.parent for c in B.cands))
        r = ranks(T, a_ids, b_ids)
        pairs = cs.match(A, B, cp)
        keys = [(A.cands[p['i']].parent, B.cands[p['j']].parent) for p in pairs]
        g = sum(k in GOOD for k in keys); w = sum(k in WRONG for k in keys)
        new = [p for p, k in zip(pairs, keys) if k not in JUDGED]
        print(f"{name:40s} MRR {np.mean(1/r):.3f}  hit@1 {np.mean(r==1):.2f}  "
              f"hit@3 {np.mean(r<=3):.2f}  median rank {np.median(r):.0f} | "
              f"matches {len(pairs):2d}: correct {g}, wrong {w}, unjudged {len(new)}")
        if new:
            tag = name.split(' (')[0].replace(' ', '_').replace('+', '')
            cv2.imwrite(f"eval_new_{tag}.jpg", cs.draw_pairs(A, B, new))
            for p in new:
                print("    unjudged:", A.cands[p['i']].tid, B.cands[p['j']].tid, f"z {p['z']:.2f}")

if __name__ == "__main__":
    if "--build" in sys.argv:
        cp = cs.CrossParams(); sp = ws.SpotParams()
        pickle.dump((cs.load_page('000126.jpg', cp, sp), cs.load_page('mstest.jpg', cp, sp)),
                    open('pages_eval.pkl', 'wb'))
    main()