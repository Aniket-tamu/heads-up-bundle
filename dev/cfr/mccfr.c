// Original lines 1-33 are outside the supplied photos.
































#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAXA 4

static int NN;
static const int32_t *CH;
static const int8_t *ACT, *STR, *NACT, *TERM;
static const int16_t *VAL;
static const int32_t *ISET;
static int NB[4];
static int64_t NI;
static double *REG, *SSUM;
static int32_t *INODE;
static int64_t ND;
static const uint16_t *BK;
static const int8_t *WIN;

int cfr_setup(int nn, const int32_t *ch, const int8_t *act, const int8_t *str, const int8_t *nact,
              const int8_t *term, const int16_t *val, const int32_t *iset, const int32_t *nb,
              int64_t ni, double *reg, double *ssum, int64_t nd, const uint16_t *bk,
              const int8_t *win) {
    NN = nn; CH = ch; ACT = act; STR = str; NACT = nact; TERM = term; VAL = val; ISET = iset;
    for (int s = 0; s < 4; s++) NB[s] = nb[s];
    NI = ni; REG = reg; SSUM = ssum; ND = nd; BK = bk; WIN = win;
    free(INODE);
    INODE = (int32_t *)malloc(sizeof(int32_t) * (size_t)ni);
    if (!INODE) return -1;
    int64_t seen = 0;
    for (int n = 0; n < nn; n++) {
        if (term[n]) continue;
        for (int b = 0; b < nb[str[n]]; b++) INODE[iset[n] + b] = n;
        seen += nb[str[n]];
    }
    return seen == ni ? 0 : -2;
}

/* ---------------------------------------------------------------- RNG */
typedef struct { uint64_t s; } Rng;

static inline uint64_t rnext(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double runif(Rng *r) { return (double)(rnext(r) >> 11) * (1.0 / 9007199254740992.0); }

/* ----------------------------------------------------------- training */
typedef struct {
    int b[2][4];        /* bucket of player (0 button, 1 big blind) per street */
    int w;              /* showdown result for the button: +1, -1, 0 */
    int p;              /* traverser */
    int prune;
    double pthr, floor, eps;
    Rng rng;
    int64_t nodes;
} TS;

/* Regret matching: returns 1 when the strategy is uniform (no positive regret). */
static inline int rmatch(const double *R, int k, double *s) {
    double sum = 0;
    for (int j = 0; j < k; j++) {
        double x = R[j] > 0 ? R[j] : 0;
        s[j] = x;
        sum += x;
    }
    if (sum > 0) {
        double inv = 1.0 / sum;
        for (int j = 0; j < k; j++) s[j] *= inv;
        return 0;
    }
    for (int j = 0; j < k; j++) s[j] = 1.0 / k;
    return 1;
}

static double trav(int n, TS *t) {
    t->nodes++;
    int tm = TERM[n];
    if (tm) {
        double v = VAL[n];
        if (tm == 3) v *= t->w;
        return t->p ? -v : v;
    }
    int a = ACT[n], k = NACT[n];
    int64_t is = ((int64_t)ISET[n] + t->b[a][STR[n]]) * MAXA;
    double *R = REG + is;
    double s[MAXA];
    int uni = rmatch(R, k, s);
    const int32_t *ch = CH + (int64_t)n * MAXA;
    if (a == t->p) {
        double va[MAXA] = {0, 0, 0, 0}, v = 0;
        int ex[MAXA] = {0, 0, 0, 0};
        int pr = t->prune && !uni;
        for (int j = 0; j < k; j++) {
            if (pr && R[j] < t->pthr && !TERM[ch[j]]) { ex[j] = 0; continue; }
            ex[j] = 1;
            va[j] = trav(ch[j], t);
            v += s[j] * va[j];
        }
        for (int j = 0; j < k; j++) {
            if (!ex[j]) continue;
            double r = R[j] + (va[j] - v);
            R[j] = r < t->floor ? t->floor : r;
        }
        return v;
    }
    double *S = SSUM + is;
    for (int j = 0; j < k; j++) S[j] += s[j];
    /* sample from q = (1 - eps) s + eps / k; the value is corrected by s/q for the
     * ancestors, regrets below are not (exploration trains off-path responses) */
    double q[MAXA], e = t->eps / k, u = runif(&t->rng);
    for (int j = 0; j < k; j++) q[j] = (1.0 - t->eps) * s[j] + e;
    int j = 0;
    double c = q[0];
    while (j < k - 1 && u >= c) { j++; c += q[j]; }
    double v = trav(ch[j], t);
    return t->eps > 0 ? v * (s[j] / q[j]) : v;
}

/* Run traversals on `threads` Hogwild threads for `seconds` (or until max_trav > 0
 * traversals). Returns the number of traversals; *nodes_out gets the nodes touched. */
int64_t cfr_run(double seconds, int64_t max_trav, int threads, uint64_t seed, int prune_on,
                double prune_prob, double prune_thr, double floor, double eps, int64_t *nodes_out) {
    int64_t total = 0, nodes = 0;
    double t0 = omp_get_wtime();
#pragma omp parallel num_threads(threads) reduction(+ : total, nodes)
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        TS t;
        memset(&t, 0, sizeof t);
        t.rng.s = seed * 0xD1B54A32D192ED03ULL + (uint64_t)(tid + 1) * 0x9E3779B97F4A7C15ULL;
        rnext(&t.rng);
        t.pthr = prune_thr;
        t.floor = floor;
        t.eps = eps;
        int64_t it = 0;
        int64_t cap = max_trav > 0 ? (max_trav + nth - 1) / nth : -1;
        for (;;) {
            if ((it & 63) == 0) {
                if (seconds > 0 && omp_get_wtime() - t0 > seconds) break;
            }
            if (cap >= 0 && it >= cap) break;
            uint64_t r = rnext(&t.rng);
            int64_t d = (int64_t)((r >> 1) % (uint64_t)ND);
            int sb = (int)(r & 1);               /* pool seat on the button */
            const uint16_t *bk = BK + d * 8;
            for (int s = 0; s < 4; s++) {
                t.b[0][s] = bk[sb * 4 + s];
                t.b[1][s] = bk[(1 - sb) * 4 + s];
            }
            t.w = sb ? -WIN[d] : WIN[d];
            t.p = (int)((it + tid) & 1);
            t.prune = prune_on && runif(&t.rng) < prune_prob;
            trav(0, &t);
            it++;
        }
        total += it;
        nodes += t.nodes;
    }
    if (nodes_out) *nodes_out = nodes;
    return total;
}

/* Multiply positive regrets by fpos, negative regrets by fneg, strategy sums by fs. */
void cfr_discount(double fpos, double fneg, double fs, int threads) {
    int64_t m = NI * MAXA;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i = 0; i < m; i++) {
        double r = REG[i];
        REG[i] = r > 0 ? r * fpos : r * fneg;
        SSUM[i] *= fs;
    }
}

/* Normalised strategy per info set (slot order); which = 0 average, 1 current.
 * Info sets with no strategy weight get uniform (avg) and mass[i] = 0. */
void cfr_strategy(int which, float *out, double *mass, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i = 0; i < NI; i++) {
        int k = NACT[INODE[i]];
        const double *src = (which ? REG : SSUM) + i * MAXA;
        double sum = 0;
        for (int j = 0; j < k; j++) sum += src[j] > 0 ? src[j] : 0;
        if (mass) mass[i] = which ? sum : (sum > 0 ? sum : 0);
        for (int j = 0; j < MAXA; j++) {
            float v = 0;
            if (j < k) v = sum > 0 ? (float)((src[j] > 0 ? src[j] : 0) / sum) : (float)(1.0 / k);
            out[i * MAXA + j] = v;
        }
    }
}

/* ----------------------------------------- best response / evaluation */
typedef struct {
    int K, p, mode, taskd;
    const float *sig;       /* [NI*4] strategy of both players */
    const float *sigp;      /* [NI*4] strategy of p in mode 2 (NULL = sig) */
    const uint16_t *bk;     /* [2][4][K] bucket per player, street, view */
    const float *w;         /* [K] showdown result for the button */
    int8_t *bra;            /* [NI] slot chosen for p's info sets (-1 = play sig) */
    double tau, margin;
} BRC;

/* Values for p of the m active views vid[] (opponent reach rch[]) at node n -> out[].
 * Views whose reach falls to tau or below are dropped (their share is negligible).
 * Subtrees above depth taskd run as OpenMP tasks; they share nothing but bra rows of
 * distinct nodes. */
static void brr(const BRC *c, int n, int m, const int32_t *vid, const float *rch, float *out, int depth) {
    const int K = c->K, p = c->p, tm = TERM[n];
    if (tm) {
        float v = (float)(p ? -VAL[n] : VAL[n]);
        if (tm == 3)
            for (int i = 0; i < m; i++) out[i] = v * c->w[vid[i]];
        else
            for (int i = 0; i < m; i++) out[i] = v;
        return;
    }
    const int a = ACT[n], kk = NACT[n], s = STR[n];
    const int32_t *ch = CH + (int64_t)n * MAXA;
    const float *sg = ((a == p && c->sigp) ? c->sigp : c->sig) + (int64_t)ISET[n] * MAXA;
    const uint16_t *ba = c->bk + ((size_t)a * 4 + s) * (size_t)K;
    const int spawn = depth < c->taskd;
    if (a == p && c->mode == 0) {
        float *vo = (float *)malloc(sizeof(float) * (size_t)kk * (size_t)m);
        if (!vo) abort();
        for (int j = 0; j < kk; j++) {
            float *dst = vo + (size_t)j * m;
            int cj = ch[j];
            if (spawn) {
#pragma omp task firstprivate(cj, dst)
                brr(c, cj, m, vid, rch, dst, depth + 1);
            } else {
                brr(c, cj, m, vid, rch, dst, depth + 1);
            }
        }
        if (spawn) {
#pragma omp taskwait
        }
        const int nb = NB[s];
        /* per bucket: A[0..3] sum r*v_j, A[4] sum r, A[5..8] sum (r*(v_j - v_sig))^2 */
        double *agg = (double *)calloc((size_t)nb * 9, sizeof(double));
        if (!agg) abort();
        for (int i = 0; i < m; i++) {
            int b = ba[vid[i]];
            double r = rch[i], *A = agg + (size_t)b * 9, vs = 0;
            for (int j = 0; j < kk; j++) vs += sg[b * MAXA + j] * vo[(size_t)j * m + i];
            A[4] += r;
            for (int j = 0; j < kk; j++) {
                double v = vo[(size_t)j * m + i], d = r * (v - vs);
                A[j] += r * v;
                A[5 + j] += d * d;
            }
        }
        int8_t *bra = c->bra + ISET[n];
        for (int b = 0; b < nb; b++) {
            double *A = agg + (size_t)b * 9;
            int best = -1;
            if (A[4] > 0) {
                best = 0;
                for (int j = 1; j < kk; j++)
                    if (A[j] > A[best]) best = j;
                double sv = 0;
                for (int j = 0; j < kk; j++) sv += sg[b * MAXA + j] * A[j];
                /* keep sig unless the gain is margin standard errors above 0 (views independent) */
                if (A[best] - sv <= c->margin * sqrt(A[5 + best]) || A[best] <= sv) best = -1;
            }
            bra[b] = (int8_t)best;
        }
        for (int i = 0; i < m; i++) {
            int b = ba[vid[i]], j = bra[b];
            if (j >= 0) {
                out[i] = vo[(size_t)j * m + i];
            } else {
                float v = 0;
                for (int jj = 0; jj < kk; jj++) v += sg[b * MAXA + jj] * vo[(size_t)jj * m + i];
                out[i] = v;
            }
        }
        free(agg);
        free(vo);
        return;
    }
    /* mixed node: the opponent, p in mode 2, or p's fixed response in mode 1 */
    const int fixed = (a == p && c->mode == 1);
    const int8_t *bra = fixed ? c->bra + ISET[n] : NULL;
    size_t blk = (size_t)m;
    char *mem = (char *)malloc(blk * (size_t)kk * 20);
    if (!mem) abort();
    int32_t *cv[MAXA], *cp[MAXA];
    float *cr[MAXA], *cw[MAXA], *co[MAXA];
    int mm[MAXA];
    for (int j = 0; j < kk; j++) {
        char *base = mem + blk * 20 * (size_t)j;
        cv[j] = (int32_t *)base;
        cp[j] = (int32_t *)(base + blk * 4);
        cr[j] = (float *)(base + blk * 8);
        cw[j] = (float *)(base + blk * 12);
        co[j] = (float *)(base + blk * 16);
        mm[j] = 0;
    }
    const float tau = (float)c->tau;
    for (int i = 0; i < m; i++) {
        int k = vid[i], b = ba[k];
        if (fixed && bra[b] >= 0) {
            int j = bra[b], t = mm[j]++;
            cv[j][t] = k; cp[j][t] = i; cr[j][t] = rch[i]; cw[j][t] = 1.0f;
            continue;
        }
        for (int j = 0; j < kk; j++) {
            float pr = sg[b * MAXA + j];
            if (pr <= 0) continue;
            float x = rch[i] * pr;
            if (x <= tau) continue;
            int t = mm[j]++;
            cv[j][t] = k; cp[j][t] = i; cr[j][t] = x; cw[j][t] = pr;
        }
    }
    for (int j = 0; j < kk; j++) {
        if (!mm[j]) continue;
        int cj = ch[j], mj = mm[j];
        int32_t *v = cv[j];
        float *r = cr[j], *o = co[j];
        if (spawn) {
#pragma omp task firstprivate(cj, mj, v, r, o)
            brr(c, cj, mj, v, r, o, depth + 1);
        } else {
            brr(c, cj, mj, v, r, o, depth + 1);
        }
    }
    if (spawn) {
#pragma omp taskwait
    }
    for (int i = 0; i < m; i++) out[i] = 0;
    for (int j = 0; j < kk; j++)
        for (int t = 0; t < mm[j]; t++) out[cp[j][t]] += cw[j][t] * co[j][t];
    free(mem);
}

/* Value for player p of every view: mode 0 best response (fills bra; an action replaces
 * sig only where its gain over sig exceeds margin * sqrt(sum of squared reaches)),
 * mode 1 fixed response bra, mode 2 sig against itself (or p playing sigp against sig,
 * when sigp is not NULL). bk is [2][4][K], w [K]. */
int cfr_br(int mode, int p, const float *sig, int K, const uint16_t *bk, const float *w,
           int8_t *bra, float *out, int threads, double tau, double margin, const float *sigp) {
    BRC c;
    memset(&c, 0, sizeof c);
    c.K = K; c.p = p; c.mode = mode; c.sig = sig; c.bk = bk; c.w = w; c.bra = bra;
    c.sigp = mode == 2 ? sigp : NULL;
    c.tau = tau; c.margin = margin;
    c.taskd = K > 400000 ? 3 : 5;    /* fewer live task buffers for big view sets */
    int32_t *vid = (int32_t *)malloc(sizeof(int32_t) * (size_t)K);
    float *rch = (float *)malloc(sizeof(float) * (size_t)K);
    if (!vid || !rch) return -1;
    for (int k = 0; k < K; k++) { vid[k] = k; rch[k] = 1.0f; }
    if (mode == 0) memset(bra, -1, (size_t)NI);
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        brr(&c, 0, K, vid, rch, out, 0);
    }
    free(vid);
    free(rch);
    return 0;
}
