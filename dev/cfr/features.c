/* Offline card-abstraction features, bit-identical to plo5bot/abstraction.py.
 *
 * Built as a shared library (dev/cfr/build.sh) and driven from Python through ctypes
 * (dev/cfr/cfeat.py). Cards are 0..35 (rank = c >> 2, suit = c & 3). PT is read from
 * plo5bot/data/pair_triple_rank.npy (uint16, 630 pair ids x 7140 triple ids, 1 + dense
 * rank, 0 = overlapping cards) and kept transposed so one triple's 630 pair values are
 * contiguous. All features are integer counts, 2 per win and 1 per tie:
 *
 *   flop  eq2  over all 378 (turn, river) runouts of the 28 live cards x MF opponents each
 *         now2 the same opponents compared on the 3-card flop
 *         sqt  sum over turn cards p of S_p^2 - sum of (runout sums)^2 containing p: unbiased
 *              28 x 4 x (54^2 - 27 x 4) x E_turn[HS_turn^2] (S_p = eq2 of the runouts holding p)
 *         sqr  sum over runouts of w_a * w_b over opponent pairs: unbiased 4 x 378 x E[HS_river^2]
 *   turn  eq2  over all 27 river cards x MT opponents each; now2 on the 4-card turn board;
 *         sq   sum over river cards of (that river's eq2)^2
 *   river eq2  exact, against all C(26, 5) = 65,780 opponent holdings (independent-set count)
 *
 * Live cards = cards not in the hand and not in the current board, ascending; the
 * opponent tables hold positions into that list (see dev/cfr/ABSTRACTION.md).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPAIRS 630
#define NTRIPLES 7140
#define NFEAT 9

static uint16_t *PTT = NULL;                 /* [NTRIPLES][NPAIRS] */
static uint8_t *FLOP_OPP = NULL;              /* [378][MF][5] */
static uint8_t *TURN_OPP = NULL;              /* [27][MT][5] */
static int MF = 0, MT = 0;

static const int T10[10][3] = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4},
                              {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}};
static const int T4[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
static const int P10[10][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2},
                              {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};

static inline int pid2(int a, int b) { return a < b ? b * (b - 1) / 2 + a : a * (a - 1) / 2 + b; }

static inline int tid3(int a, int b, int c) {
    int t;
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    return c * (c - 1) * (c - 2) / 6 + b * (b - 1) / 2 + a;
}

/* Read a .npy header, check dtype/order/shape, leave the file at the data; 0 on success. */
static int npy_check(FILE *f, const char *descr, const char *shape) {
    unsigned char pre[12];
    long hlen;
    if (fread(pre, 1, 10, f) != 10 || memcmp(pre, "\x93NUMPY", 6) != 0) return -1;
    if (pre[6] == 1) {
        hlen = pre[8] | (pre[9] << 8);
    } else {
        if (fread(pre + 10, 1, 2, f) != 2) return -1;
        hlen = pre[8] | (pre[9] << 8) | ((long)pre[10] << 16) | ((long)pre[11] << 24);
    }
    char *h = malloc(hlen + 1);
    if (!h || fread(h, 1, hlen, f) != (size_t)hlen) { free(h); return -1; }
    h[hlen] = 0;
    int ok = strstr(h, descr) && strstr(h, "'fortran_order': False") && strstr(h, shape);
    free(h);
    return ok ? 0 : -2;
}

static uint8_t *copy_table(const uint8_t *src, size_t n) {
    uint8_t *dst = malloc(n);
    if (dst) memcpy(dst, src, n);
    return dst;
}

/* Load PT and the opponent tables (positions, uint8). Returns 0 on success. */
int feat_init(const char *pt_path, const uint8_t *flop, int mf, const uint8_t *turn, int mt) {
    if (mf < 1 || mf > 64 || mt < 1) return -6; /* flop_feats keeps 64 results per runout */
    FILE *f = fopen(pt_path, "rb");
    if (!f) return -1;
    if (npy_check(f, "<u2", "(630, 7140)") != 0) { fclose(f); return -2; }
    uint16_t *pt = malloc((size_t)NPAIRS * NTRIPLES * sizeof(uint16_t));
    if (!pt || fread(pt, sizeof(uint16_t), (size_t)NPAIRS * NTRIPLES, f) != (size_t)NPAIRS * NTRIPLES) {
        fclose(f); free(pt); return -3;
    }
    fclose(f);
    free(PTT);
    PTT = malloc((size_t)NPAIRS * NTRIPLES * sizeof(uint16_t));
    if (!PTT) { free(pt); return -4; }
    for (int p = 0; p < NPAIRS; p++)
        for (int t = 0; t < NTRIPLES; t++) PTT[(size_t)t * NPAIRS + p] = pt[(size_t)p * NTRIPLES + t];
    free(pt);
    free(FLOP_OPP); free(TURN_OPP);
    FLOP_OPP = copy_table(flop, (size_t)378 * mf * 5);
    TURN_OPP = copy_table(turn, (size_t)27 * mt * 5);
    MF = mf; MT = mt;
    return (FLOP_OPP && TURN_OPP) ? 0 : -5;
}

/* pv[p] = best (1 + dense rank) of pair p with the 5-card board b. */
static void board_vals(const int *b, uint16_t *restrict pv) {
    const uint16_t *r[10];
    for (int k = 0; k < 10; k++) r[k] = PTT + (size_t)tid3(b[T10[k][0]], b[T10[k][1]], b[T10[k][2]]) * NPAIRS;
    for (int p = 0; p < NPAIRS; p++) {
        uint16_t m = r[0][p];
        for (int k = 1; k < 10; k++) m = r[k][p] > m ? r[k][p] : m;
        pv[p] = m;
    }
}

static inline uint16_t hand_max(const uint16_t *pv, const int *pairs10) {
    uint16_t m = 0;
    for (int k = 0; k < 10; k++) m = pv[pairs10[k]] > m ? pv[pairs10[k]] : m;
    return m;
}

static inline void pairs_of(const int *c5, int *pairs10) {
    for (int k = 0; k < 10; k++) pairs10[k] = pid2(c5[P10[k][0]], c5[P10[k][1]]);
}

static int make_live(const int *hole, const int *board, int nb, int *live) {
    int used[36] = {0}, n = 0;
    for (int k = 0; k < 5; k++) used[hole[k]] = 1;
    for (int k = 0; k < nb; k++) used[board[k]] = 1;
    for (int c = 0; c < 36; c++)
        if (!used[c]) live[n++] = c;
    return n;
}

static inline int32_t score(uint16_t h, uint16_t o) { return h > o ? 2 : (h == o ? 1 : 0); }

static void flop_feats(const int *hole, const int *b, int32_t *eq2_out, int32_t *now2_out,
                       int32_t *sqt_out, int32_t *sqr_out) {
    int live[36], hp[10], op[10], o[5], b5[5] = {b[0], b[1], b[2], 0, 0};
    int32_t S[28] = {0}, T[28] = {0}, w[64];
    uint16_t pv[NPAIRS];
    make_live(hole, b, 3, live); /* 28 cards */
    pairs_of(hole, hp);
    const uint16_t *rf = PTT + (size_t)tid3(b[0], b[1], b[2]) * NPAIRS;
    uint16_t hf = hand_max(rf, hp);
    int32_t eq2 = 0, now2 = 0, sqr = 0;
    int u = 0;
    for (int i = 0; i < 28; i++) {
        for (int j = i + 1; j < 28; j++, u++) {
            b5[3] = live[i];
            b5[4] = live[j];
            board_vals(b5, pv);
            uint16_t hv = hand_max(pv, hp);
            int32_t su = 0, qu = 0;
            for (int m = 0; m < MF; m++) {
                const uint8_t *pos = FLOP_OPP + ((size_t)u * MF + m) * 5;
                for (int k = 0; k < 5; k++) o[k] = live[pos[k]];
                pairs_of(o, op);
                w[m] = score(hv, hand_max(pv, op));
                su += w[m];
                qu += w[m] * w[m];
                now2 += score(hf, hand_max(rf, op));
            }
            eq2 += su;
            sqr += (su * su - qu) / 2;  /* sum of w_a * w_b over opponent pairs */
            S[i] += su; S[j] += su;
            T[i] += su * su; T[j] += su * su;
        }
    }
    int32_t sqt = 0;
    for (int p = 0; p < 28; p++) sqt += S[p] * S[p] - T[p];
    *eq2_out = eq2;
    *now2_out = now2;
    *sqt_out = sqt;
    *sqr_out = sqr;
}

static void turn_feats(const int *hole, const int *b, int32_t *eq2_out, int32_t *now2_out, int32_t *sq_out) {
    int live[36], hp[10], op[10], o[5], b5[5] = {b[0], b[1], b[2], b[3], 0};
    uint16_t pv[NPAIRS], base4[NPAIRS];
    make_live(hole, b, 4, live); /* 27 cards */
    pairs_of(hole, hp);
    const uint16_t *r4[4];
    for (int k = 0; k < 4; k++) r4[k] = PTT + (size_t)tid3(b[T4[k][0]], b[T4[k][1]], b[T4[k][2]]) * NPAIRS;
    for (int p = 0; p < NPAIRS; p++) {
        uint16_t m = r4[0][p];
        for (int k = 1; k < 4; k++) m = r4[k][p] > m ? r4[k][p] : m;
        base4[p] = m;
    }
    uint16_t h4 = hand_max(base4, hp);
    int32_t eq2 = 0, now2 = 0, sq = 0;
    for (int j = 0; j < 27; j++) {
        b5[4] = live[j];
        board_vals(b5, pv);
        uint16_t hv = hand_max(pv, hp);
        int32_t w2 = 0;
        for (int m = 0; m < MT; m++) {
            const uint8_t *pos = TURN_OPP + ((size_t)j * MT + m) * 5;
            for (int k = 0; k < 5; k++) o[k] = live[pos[k]];
            pairs_of(o, op);
            w2 += score(hv, hand_max(pv, op));
            now2 += score(h4, hand_max(base4, op));
        }
        eq2 += w2;
        sq += w2 * w2;
    }
    *eq2_out = eq2;
    *now2_out = now2;
    *sq_out = sq;
}

#define ALL26 ((1u << 26) - 1)
#define ABOVE(v) (ALL26 & ~((1u << (v)) - 1)) /* bits of positions above v */

/* Number of 5-subsets of 26 positions with no adjacent pair; na[v] = non-neighbours of v.
 * Each set is counted once: its 4 lowest positions plus one higher common non-neighbour. */
static int32_t count_independent5(const uint32_t *na) {
    int32_t cnt = 0;
    for (int i = 0; i < 26; i++) {
        uint32_t mj = na[i] & ABOVE(i);
        while (mj) {
            int j = __builtin_ctz(mj);
            mj &= mj - 1;
            uint32_t mij = na[i] & na[j] & ABOVE(j), mk = mij;
            while (mk) {
                int k = __builtin_ctz(mk);
                mk &= mk - 1;
                uint32_t mijk = mij & na[k] & ABOVE(k), ml = mijk;
                while (ml) {
                    int l = __builtin_ctz(ml);
                    ml &= ml - 1;
                    cnt += __builtin_popcount(mijk & na[l] & ABOVE(l));
                }
            }
        }
    }
    return cnt;
}

/* Exact river eq2: an opponent holding is worth < hv iff none of its pairs is worth >= hv. */
static int32_t river_feat(const int *hole, const int *b, const uint16_t *pv) {
    int live[36], hp[10], tie = 0;
    uint32_t na_ge[26], na_gt[26];
    make_live(hole, b, 5, live); /* 26 cards */
    pairs_of(hole, hp);
    uint16_t hv = hand_max(pv, hp);
    for (int v = 0; v < 26; v++) na_ge[v] = na_gt[v] = ALL26 & ~(1u << v);
    for (int j = 1; j < 26; j++) {
        for (int i = 0; i < j; i++) {
            uint16_t x = pv[pid2(live[i], live[j])];
            if (x >= hv) { na_ge[i] &= ~(1u << j); na_ge[j] &= ~(1u << i); }
            if (x > hv) { na_gt[i] &= ~(1u << j); na_gt[j] &= ~(1u << i); }
            tie |= x == hv;
        }
    }
    int32_t below = count_independent5(na_ge);
    return tie ? below + count_independent5(na_gt) : 2 * below;
}

/* cards: n x 15 (seat 0 hole, seat 1 hole, board in deal order: flop 3, turn, river).
 * out: n x 2 x NFEAT int32 (column 0, the preflop class, is left for the caller: -1).
 * winner: n int8, +1 seat 0 wins the showdown, -1 seat 1, 0 tie. */
void feat_batch(const uint8_t *cards, int64_t n, int32_t *out, int8_t *winner, int nthreads) {
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
    for (int64_t d = 0; d < n; d++) {
        const uint8_t *c = cards + d * 15;
        int b[5], hole[5], hp[10];
        uint16_t pvfull[NPAIRS], hv[2];
        for (int k = 0; k < 5; k++) b[k] = c[10 + k];
        board_vals(b, pvfull);
        for (int s = 0; s < 2; s++) {
            for (int k = 0; k < 5; k++) hole[k] = c[5 * s + k];
            int32_t *o = out + ((size_t)d * 2 + s) * NFEAT;
            o[0] = -1;
            flop_feats(hole, b, &o[1], &o[2], &o[7], &o[8]);
            turn_feats(hole, b, &o[3], &o[4], &o[5]);
            o[6] = river_feat(hole, b, pvfull);
            pairs_of(hole, hp);
            hv[s] = hand_max(pvfull, hp);
        }
        winner[d] = hv[0] > hv[1] ? 1 : (hv[0] < hv[1] ? -1 : 0);
    }
}

/* Showdown values only (1 + dense rank of each seat's best hand), n x 2 uint16. */
void showdown_values(const uint8_t *cards, int64_t n, uint16_t *out, int nthreads) {
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int64_t d = 0; d < n; d++) {
        const uint8_t *c = cards + d * 15;
        int b[5], hole[5], hp[10];
        uint16_t pv[NPAIRS];
        for (int k = 0; k < 5; k++) b[k] = c[10 + k];
        board_vals(b, pv);
        for (int s = 0; s < 2; s++) {
            for (int k = 0; k < 5; k++) hole[k] = c[5 * s + k];
            pairs_of(hole, hp);
            out[d * 2 + s] = hand_max(pv, hp);
        }
    }
}
