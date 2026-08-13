/* h3_tokenizer.c - BPE tokenizer for the CUDA/Linux build (feat/cuda).
 *
 * Port of h3_tokenizer.m (Metal/Objective-C) to portable C11. The Metal
 * implementation is preserved untouched; this file is the Linux/CUDA twin.
 * It parses the released tokenizer.json (BPE, NFC normalizer, byte-level
 * vocabulary) and implements prompt encode / token decode with the same
 * byte-level BPE merge algorithm as the reference.
 *
 * ICU is linked as the runtime library only (no -dev headers installed on the
 * DGX Spark): we declare the versioned ABI symbols ourselves. This keeps the
 * character categories (letter/number/space) and NFC normalization bit-equal
 * to the Foundation/ICU path used by the Metal reference.
 *   ponytail: versioned symbols (u_charType_74, ...) pin the build to ICU 74;
 *   that is fine because the Spark build gate is the only supported target.
 */
#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- ICU runtime ABI (no headers installed) ----------------------------- */
typedef int32_t UChar32;
typedef int32_t UErrorCode;
typedef uint16_t UChar;
typedef struct UNormalizer2 UNormalizer2;

extern int8_t  u_charType_74(UChar32 c);
extern int     u_isUWhiteSpace_74(UChar32 c);
extern const UNormalizer2 *unorm2_getNFCInstance_74(UErrorCode *pErrorCode);
extern int32_t unorm2_normalize_74(const UNormalizer2 *norm2,
                                   const UChar *src, int32_t srcLength,
                                   UChar *dest, int32_t destCapacity,
                                   UErrorCode *pErrorCode);

/* ---- tokenizer implementation ------------------------------------------ */
typedef struct h3_tokenizer_impl {
    /* symbol -> id */
    struct tok_map *vocab;
    char **inverse_vocab;      /* id -> symbol (NULL if absent) */
    size_t max_id;

    struct tok_map *merge_ranks;      /* pair key -> rank */
    struct tok_map *added_tokens;     /* content -> id */
    char **inverse_added;      /* id -> content (NULL if absent) */
    char **added_alternatives; /* sorted content, longest first */
    size_t added_count;

    char **byte_encoder;       /* 256 byte -> codepoint string */
    int16_t byte_decoder[324];

    /* bpe cache: encoded piece -> id list */
    struct tok_map *bpe_cache;
    char **cache_ids;          /* value: packed uint32 ids */
    size_t *cache_counts;
    size_t cache_cap;
} impl_t;

static void tok_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
}

/* ---- UTF-8 <-> UTF-16 --------------------------------------------------- */
static size_t utf8_to_utf16(const char *utf8, UChar *out, size_t cap) {
    const unsigned char *p = (const unsigned char *)utf8;
    size_t n = 0;
    while (*p) {
        uint32_t cp;
        size_t len;
        unsigned c = *p;
        if (c < 0x80)      { cp = c; len = 1; }
        else if (c < 0xC0) { cp = 0xFFFD; len = 1; }
        else if (c < 0xE0) { cp = ((c & 0x1F) << 6) | (p[1] & 0x3F); len = 2; }
        else if (c < 0xF0) { cp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); len = 3; }
        else               { cp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); len = 4; }
        if (cp >= 0x10000) {
            if (n + 2 > cap) break;
            cp -= 0x10000;
            out[n++] = (UChar)(0xD800 + (cp >> 10));
            out[n++] = (UChar)(0xDC00 + (cp & 0x3FF));
        } else {
            if (n + 1 > cap) break;
            out[n++] = (UChar)cp;
        }
        p += len;
    }
    return n;
}

static void utf16_to_utf8(const UChar *s, size_t len, char *out, size_t cap) {
    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        uint32_t cp = s[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len &&
            s[i+1] >= 0xDC00 && s[i+1] <= 0xDFFF) {
            cp = 0x10000 + ((uint32_t)(s[i] - 0xD800) << 10) + (s[i+1] - 0xDC00);
            i++;
        }
        if (cp < 0x80)      { if (n + 1 > cap) break; out[n++] = (char)cp; }
        else if (cp < 0x800) { if (n + 2 > cap) break; out[n++] = (char)(0xC0 | (cp >> 6)); out[n++] = (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { if (n + 3 > cap) break; out[n++] = (char)(0xE0 | (cp >> 12)); out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[n++] = (char)(0x80 | (cp & 0x3F)); }
        else { if (n + 4 > cap) break; out[n++] = (char)(0xF0 | (cp >> 18)); out[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); out[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[n++] = (char)(0x80 | (cp & 0x3F)); }
    }
    if (n < cap) out[n] = '\0';
}

/* ---- string-keyed hash map --------------------------------------------- */
typedef struct tok_map_entry {
    char *key;
    uint32_t value;
    struct tok_map_entry *next;
} tok_map_entry;

typedef struct tok_map {
    size_t size;
    tok_map_entry **slots;
} tok_map;

static tok_map *map_new(size_t size) {
    tok_map *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->size = size ? size : 1;
    m->slots = calloc(m->size, sizeof(*m->slots));
    if (!m->slots) { free(m); return NULL; }
    return m;
}

static void map_put(tok_map *m, const char *key, size_t key_len, uint32_t value) {
    char *dup = malloc(key_len + 1);
    memcpy(dup, key, key_len);
    dup[key_len] = '\0';
    uint32_t h = 5381;
    for (size_t i = 0; i < key_len; i++) h = h * 33 + (unsigned char)key[i];
    tok_map_entry *e = malloc(sizeof(*e));
    e->key = dup; e->value = value; e->next = m->slots[h % m->size];
    m->slots[h % m->size] = e;
}

static int map_get(const tok_map *m, const char *key, size_t key_len,
                   uint32_t *value) {
    uint32_t h = 5381;
    for (size_t i = 0; i < key_len; i++) h = h * 33 + (unsigned char)key[i];
    for (tok_map_entry *e = m->slots[h % m->size]; e; e = e->next) {
        if (strlen(e->key) == key_len && memcmp(e->key, key, key_len) == 0) {
            if (value) *value = e->value;
            return 1;
        }
    }
    return 0;
}

static void map_free(tok_map *m) {
    if (!m) return;
    for (size_t i = 0; i < m->size; i++) {
        tok_map_entry *e = m->slots[i];
        while (e) { tok_map_entry *n = e->next; free(e->key); free(e); e = n; }
    }
    free(m->slots);
    free(m);
}

/* ---- codepoint view over a UTF-16 buffer -------------------------------- */
typedef struct {
    UChar *text;
    size_t length;       /* UTF-16 code-unit count */
    uint32_t *value;
    size_t *location;    /* UTF-16 index of codepoint start */
    size_t *units;       /* UTF-16 units of codepoint (1 or 2) */
    size_t count;
} codepoints;

static int cp_build(codepoints *cp, UChar *text, size_t length) {
    cp->text = text;
    cp->length = length;
    cp->count = 0;
    cp->value = malloc((length ? length : 1) * sizeof(uint32_t));
    cp->location = malloc((length ? length : 1) * sizeof(size_t));
    cp->units = malloc((length ? length : 1) * sizeof(size_t));
    if (!cp->value || !cp->location || !cp->units) return 0;
    size_t used = 0;
    for (size_t i = 0; i < length;) {
        uint32_t v = text[i];
        size_t u = 1;
        if (text[i] >= 0xD800 && text[i] <= 0xDBFF && i + 1 < length &&
            text[i+1] >= 0xDC00 && text[i+1] <= 0xDFFF) {
            v = 0x10000 + (((uint32_t)text[i] - 0xD800) << 10) + (text[i+1] - 0xDC00);
            u = 2;
        }
        cp->value[used] = v;
        cp->location[used] = i;
        cp->units[used] = u;
        used++;
        i += u;
    }
    cp->count = used;
    return 1;
}

static void cp_free(codepoints *cp) {
    free(cp->value); free(cp->location); free(cp->units);
}

/* ---- character classification (ICU) ------------------------------------- */
static int cp_letter(uint32_t v) {
    int8_t cat = u_charType_74((UChar32)v);
    return cat == 1 || cat == 2 || cat == 3 || cat == 4 || cat == 5;
}

static int cp_number(uint32_t v) {
    int8_t cat = u_charType_74((UChar32)v);
    return cat == 9 || cat == 10 || cat == 11;
}

static int cp_space(uint32_t v) {
    return u_isUWhiteSpace_74((UChar32)v) || (v >= 0x1c && v <= 0x1f);
}

/* slice codepoints [start..stop) -> UTF-8 string (caller frees) */
static char *cp_slice(const codepoints *cp, size_t start, size_t stop) {
    size_t loc = cp->location[start];
    size_t end = cp->location[stop - 1] + cp->units[stop - 1];
    char *out = malloc(end * 3 + 1);
    if (!out) return NULL;
    utf16_to_utf8(cp->text + loc, end - loc, out, end * 3 + 1);
    return out;
}

static size_t cp_contraction(const codepoints *cp, size_t index) {
    static const char *values[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (cp->value[index] != '\'') return 0;
    for (size_t item = 0; item < sizeof(values)/sizeof(values[0]); item++) {
        size_t length = strlen(values[item]);
        if (index + length > cp->count) continue;
        int matches = 1;
        for (size_t off = 1; off < length; off++) {
            uint32_t got = cp->value[index + off];
            if (got >= 'A' && got <= 'Z') got += 'a' - 'A';
            if (got != (unsigned char)values[item][off]) matches = 0;
        }
        if (matches) return length;
    }
    return 0;
}

/* pre-tokenize: port of Metal h3_pretokenize (matches the GPT2-style Split
 * regex in tokenizer.json). Returns array of UTF-8 piece strings. */
static char **pretokenize(UChar *text, size_t length, size_t *out_count,
                          char *error, size_t error_size) {
    codepoints cp;
    if (!cp_build(&cp, text, length)) { tok_error(error, error_size, "oom"); return NULL; }
    char **pieces = NULL;
    size_t npieces = 0, cap = 0;
    size_t index = 0;
    int ok = 1;
    while (index < cp.count) {
        size_t contraction = cp_contraction(&cp, index);
        if (contraction) {
            char *s = cp_slice(&cp, index, index + contraction);
            if (!s) { ok = 0; break; }
            if (npieces == cap) { cap = cap ? cap * 2 : 16; pieces = realloc(pieces, cap * sizeof(char*)); }
            pieces[npieces++] = s;
            index += contraction;
            continue;
        }
        uint32_t value = cp.value[index];
        ptrdiff_t letter_start = (ptrdiff_t)index;
        if (cp_letter(value)) {
            /* already at first letter */
        } else if (value != '\r' && value != '\n' && !cp_number(value) &&
                   index + 1 < cp.count && cp_letter(cp.value[index + 1])) {
            letter_start++;
        } else {
            letter_start = -1;
        }
        if (letter_start >= 0) {
            size_t stop = (size_t)letter_start;
            while (stop < cp.count && cp_letter(cp.value[stop])) stop++;
            char *s = cp_slice(&cp, index, stop);
            if (!s) { ok = 0; break; }
            if (npieces == cap) { cap = cap ? cap * 2 : 16; pieces = realloc(pieces, cap * sizeof(char*)); }
            pieces[npieces++] = s;
            index = stop;
            continue;
        }
        if (cp_number(value)) {
            char *s = cp_slice(&cp, index, index + 1);
            if (!s) { ok = 0; break; }
            if (npieces == cap) { cap = cap ? cap * 2 : 16; pieces = realloc(pieces, cap * sizeof(char*)); }
            pieces[npieces++] = s;
            index++;
            continue;
        }
        size_t punct_start = index +
            (value == ' ' && index + 1 < cp.count &&
             !cp_space(cp.value[index + 1]) &&
             !cp_letter(cp.value[index + 1]) &&
             !cp_number(cp.value[index + 1]));
        size_t stop = punct_start;
        while (stop < cp.count && !cp_space(cp.value[stop]) &&
               !cp_letter(cp.value[stop]) && !cp_number(cp.value[stop])) stop++;
        if (stop > punct_start) {
            while (stop < cp.count &&
                   (cp.value[stop] == '\r' || cp.value[stop] == '\n')) stop++;
            char *s = cp_slice(&cp, index, stop);
            if (!s) { ok = 0; break; }
            if (npieces == cap) { cap = cap ? cap * 2 : 16; pieces = realloc(pieces, cap * sizeof(char*)); }
            pieces[npieces++] = s;
            index = stop;
            continue;
        }
        if (cp_space(value)) {
            size_t ws_end = index + 1;
            while (ws_end < cp.count && cp_space(cp.value[ws_end])) ws_end++;
            ptrdiff_t newline_end = -1;
            for (size_t cursor = index; cursor < ws_end; cursor++) {
                if (cp.value[cursor] == '\r' || cp.value[cursor] == '\n')
                    newline_end = (ptrdiff_t)cursor + 1;
            }
            size_t piece_end;
            if (newline_end >= 0) piece_end = (size_t)newline_end;
            else if (ws_end == cp.count) piece_end = ws_end;
            else if (ws_end - index > 1) piece_end = ws_end - 1;
            else piece_end = index + 1;
            char *s = cp_slice(&cp, index, piece_end);
            if (!s) { ok = 0; break; }
            if (npieces == cap) { cap = cap ? cap * 2 : 16; pieces = realloc(pieces, cap * sizeof(char*)); }
            pieces[npieces++] = s;
            index = piece_end;
            continue;
        }
        ok = 0;
        break;
    }
    cp_free(&cp);
    if (!ok) {
        for (size_t i = 0; i < npieces; i++) free(pieces[i]);
        free(pieces);
        tok_error(error, error_size, "unable to pre-tokenize input");
        return NULL;
    }
    *out_count = npieces;
    return pieces;
}

/* ---- BPE ---------------------------------------------------------------- */
/* pair key: left + UTF-8(U+FFFF) separator + right */
static char *pair_key_bytes(const char *left, const char *right, size_t *out_len) {
    size_t l = strlen(left), r = strlen(right);
    char *k = malloc(l + 3 + r);
    memcpy(k, left, l);
    k[l] = (char)0xEF; k[l+1] = (char)0xBF; k[l+2] = (char)0xBF;
    memcpy(k + l + 3, right, r);
    *out_len = l + 3 + r;
    return k;
}

/* BPE on one piece -> array of ids (caller frees ids). */
static int bpe_encode(impl_t *t, const char *piece,
                      uint32_t **ids_out, size_t *count_out,
                      char *error, size_t error_size) {
    /* byte-encode: one symbol per byte (byteEncoder[byte]) */
    const unsigned char *b = (const unsigned char *)piece;
    size_t n = 0;
    for (size_t i = 0; b[i]; i++) n++;
    char **syms = malloc((n ? n : 1) * sizeof(char*));
    for (size_t i = 0; i < n; i++) syms[i] = strdup(t->byte_encoder[b[i]]);
    size_t nsyms = n;

    while (nsyms > 1) {
        uint32_t best_rank = UINT32_MAX;
        size_t best = 0;
        int found = 0;
        for (size_t i = 0; i + 1 < nsyms; i++) {
            size_t kl;
            char *k = pair_key_bytes(syms[i], syms[i+1], &kl);
            uint32_t rank;
            if (map_get(t->merge_ranks, k, kl, &rank) && rank < best_rank) {
                best_rank = rank; best = i; found = 1;
            }
            free(k);
        }
        if (!found) break;
        char *left = syms[best];
        char *right = syms[best + 1];
        char **merged = malloc((nsyms ? nsyms : 1) * sizeof(char*));
        size_t nmerged = 0;
        for (size_t i = 0; i < nsyms;) {
            if (i + 1 < nsyms && strcmp(syms[i], left) == 0 &&
                strcmp(syms[i+1], right) == 0) {
                size_t l = strlen(left), r = strlen(right);
                char *joined = malloc(l + r + 1);
                memcpy(joined, left, l); memcpy(joined + l, right, r); joined[l+r] = '\0';
                merged[nmerged++] = joined;
                i += 2;
            } else {
                merged[nmerged++] = strdup(syms[i]);
                i++;
            }
        }
        for (size_t i = 0; i < nsyms; i++) free(syms[i]);
        free(syms);
        syms = merged;
        nsyms = nmerged;
    }

    uint32_t *ids = malloc((nsyms ? nsyms : 1) * sizeof(uint32_t));
    for (size_t i = 0; i < nsyms; i++) {
        uint32_t id;
        if (!map_get(t->vocab, syms[i], strlen(syms[i]), &id)) {
            char msg[160];
            snprintf(msg, sizeof(msg), "BPE symbol is absent from vocabulary: %s", syms[i]);
            for (size_t j = 0; j < nsyms; j++) free(syms[j]);
            free(syms); free(ids);
            tok_error(error, error_size, msg);
            return 0;
        }
        ids[i] = id;
    }
    for (size_t i = 0; i < nsyms; i++) free(syms[i]);
    free(syms);
    *ids_out = ids;
    *count_out = nsyms;
    return 1;
}

/* encode a plain (no added tokens) segment: pretokenize then BPE each piece */
static int encode_plain(impl_t *t, const char *utf8_piece, uint32_t **ids,
                        size_t *count, char *error, size_t error_size) {
    UChar *buf = malloc((strlen(utf8_piece) + 1) * 2 + 2);
    if (!buf) { tok_error(error, error_size, "oom"); return 0; }
    size_t ulen = utf8_to_utf16(utf8_piece, buf, (strlen(utf8_piece) + 1) * 2 + 2);

    /* NFC normalize */
    UErrorCode ec = 0;
    const UNormalizer2 *nfc = unorm2_getNFCInstance_74(&ec);
    UChar *norm = malloc((ulen * 3 + 2) * sizeof(UChar));
    size_t nlen = (size_t)unorm2_normalize_74(nfc, buf, (int32_t)ulen, norm,
                                              (int32_t)(ulen * 3 + 2), &ec);
    free(buf);
    if (ec < 0) { free(norm); tok_error(error, error_size, "NFC normalization failed"); return 0; }

    size_t npieces;
    char **pieces = pretokenize(norm, nlen, &npieces, error, error_size);
    if (!pieces) { free(norm); return 0; }

    uint32_t *out = NULL;
    size_t nout = 0, cap = 0;
    for (size_t i = 0; i < npieces; i++) {
        uint32_t *pids2; size_t cnt;
        if (!bpe_encode(t, pieces[i], &pids2, &cnt, error, error_size)) {
            free(pieces[i]); free(pieces); free(norm);
            free(out);
            return 0;
        }
        free(pieces[i]);
        if (nout + cnt > cap) {
            cap = (nout + cnt) * 2;
            out = realloc(out, cap * sizeof(uint32_t));
        }
        memcpy(out + nout, pids2, cnt * sizeof(uint32_t));
        nout += cnt;
        free(pids2);
    }
    free(pieces);
    free(norm);
    *ids = out;
    *count = nout;
    return 1;
}

/* ---- added-token matching (longest first, then leftmost, then longest) ---- */
static int added_match(impl_t *t, const UChar *text, size_t length,
                       size_t search_start, size_t *match_loc, size_t *match_len,
                       uint32_t *token_id) {
    int found = 0;
    for (size_t a = 0; a < t->added_count; a++) {
        const char *cand = t->added_alternatives[a];
        size_t clen = strlen(cand);
        /* convert candidate to UTF-16 and search */
        UChar *cbuf = malloc(clen * 2 + 2);
        size_t culen = utf8_to_utf16(cand, cbuf, clen * 2 + 2);
        /* naive search from search_start */
        size_t best = SIZE_MAX;
        for (size_t i = search_start; i + culen <= length; i++) {
            int eq = 1;
            for (size_t j = 0; j < culen; j++) if (text[i+j] != cbuf[j]) { eq = 0; break; }
            if (eq) { best = i; break; }
        }
        free(cbuf);
        if (best == SIZE_MAX) continue;
        if (!found || best < *match_loc ||
            (best == *match_loc && culen > *match_len)) {
            *match_loc = best;
            *match_len = culen;
            uint32_t id;
            map_get(t->added_tokens, cand, clen, &id);
            *token_id = id;
            found = 1;
        }
    }
    return found;
}

/* ---- load ---------------------------------------------------------------- */
/* Minimal recursive JSON parser over a NUL-terminated buffer. */
typedef struct { const char *p; } jc;

static void jws(jc *c) {
    while (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r') c->p++;
}
static int jtake(jc *c, char expect) {
    jws(c);
    if (*c->p == expect) { c->p++; return 1; }
    return 0;
}
static char *jstring(jc *c, char *error, size_t es) {
    jws(c);
    if (*c->p != '"') { tok_error(error, es, "expected JSON string"); return NULL; }
    c->p++;
    size_t cap = 16;
    char *out = malloc(cap);
    if (!out) { tok_error(error, es, "oom"); return NULL; }
    size_t n = 0;
    for (;;) {
        char ch = *c->p++;
        if (ch == '"') break;
        if (ch == '\\') {
            char e = *c->p++;
            if (e == 'n') ch = '\n';
            else if (e == 't') ch = '\t';
            else if (e == 'r') ch = '\r';
            else if (e == 'f') ch = '\f';
            else if (e == 'b') ch = '\b';
            else if (e == 'u') {
                unsigned v = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *c->p++;
                    v = v * 16 + (h <= '9' ? h - '0' : (h <= 'F' ? h - 'A' + 10 : h - 'a' + 10));
                }
                uint32_t cp = v;
                char tmp[8];
                utf16_to_utf8((UChar[]){ (UChar)cp }, 1, tmp, sizeof(tmp));
                size_t tl = strlen(tmp);
                if (n + tl + 1 > cap) { cap = n + tl + 2; out = realloc(out, cap); }
                memcpy(out + n, tmp, tl); n += tl;
                continue;
            } else if (e == '\\') ch = '\\';
            else ch = e;
        }
        if (n + 1 >= cap) { cap = n + 2; out = realloc(out, cap); }
        out[n++] = ch;
        out[n] = '\0';
    }
    out[n] = '\0';
    return out;
}
static void jskip(jc *c) {
    jws(c);
    if (*c->p == '{') {
        c->p++; jws(c);
        while (*c->p && *c->p != '}') {
            jws(c);
            if (*c->p == '"') { char *s = jstring(c, NULL, 0); free(s); }
            jws(c);
            if (*c->p == ':') c->p++;
            jskip(c);
            jws(c);
            if (*c->p == ',') c->p++;
        }
        if (*c->p == '}') c->p++;
    } else if (*c->p == '[') {
        c->p++; jws(c);
        while (*c->p && *c->p != ']') { jskip(c); jws(c); if (*c->p == ',') c->p++; }
        if (*c->p == ']') c->p++;
    } else if (*c->p == '"') { char *s = jstring(c, NULL, 0); free(s); }
    else if (*c->p == 't') { c->p += 4; }
    else if (*c->p == 'f') { c->p += 5; }
    else if (*c->p == 'n') { c->p += 4; }
    else { while (*c->p && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ':') c->p++; }
}

typedef struct { impl_t *t; char *error; size_t es; int failed; } load_ctx;

static void load_vocab(impl_t *t, jc *c, load_ctx *lc) {
    /* object { "symbol": id } */
    jws(c);
    if (!jtake(c, '{')) { lc->failed = 1; tok_error(lc->error, lc->es, "bad vocab"); return; }
    jws(c);
    while (*c->p && *c->p != '}') {
        char *sym = jstring(c, lc->error, lc->es);
        if (!sym) { lc->failed = 1; return; }
        jws(c);
        if (!jtake(c, ':')) { free(sym); lc->failed = 1; tok_error(lc->error, lc->es, "bad vocab ':'"); return; }
        jws(c);
        /* id number */
        uint32_t id = (uint32_t)strtoul(c->p, (char**)&c->p, 10);
        jskip(c);
        map_put(t->vocab, sym, strlen(sym), id);
        if (id > t->max_id) t->max_id = id;
        free(sym);
        jws(c);
        if (*c->p == ',') c->p++;
        jws(c);
    }
    if (*c->p == '}') c->p++;
}

static void load_merges(impl_t *t, jc *c, load_ctx *lc) {
    jws(c);
    if (!jtake(c, '[')) { lc->failed = 1; tok_error(lc->error, lc->es, "bad merges"); return; }
    jws(c);
    uint32_t rank = 0;
    while (*c->p && *c->p != ']') {
        char *left = NULL, *right = NULL;
        int owns_sep = 0;
        jws(c);
        if (*c->p == '"') {
            /* string form "left right" */
            char *s = jstring(c, lc->error, lc->es);
            if (!s) { lc->failed = 1; return; }
            char *sp = strchr(s, ' ');
            if (sp) { *sp = '\0'; left = s; right = sp + 1; owns_sep = 1; }
            else { free(s); lc->failed = 1; tok_error(lc->error, lc->es, "merge no space"); return; }
        } else if (*c->p == '[') {
            c->p++; jws(c);
            left = jstring(c, lc->error, lc->es);
            jws(c);
            if (*c->p == ',') c->p++;
            jws(c);
            right = jstring(c, lc->error, lc->es);
            jws(c);
            if (*c->p == ']') c->p++;
        } else {
            lc->failed = 1; tok_error(lc->error, lc->es, "invalid merge"); return;
        }
        if (!left || !right) { lc->failed = 1; return; }
        size_t kl;
        char *k = pair_key_bytes(left, right, &kl);
        map_put(t->merge_ranks, k, kl, rank++);
        free(k);
        if (owns_sep) {
            free(left);   /* frees the whole s allocation */
        } else {
            free(left); free(right);
        }
        jws(c);
        if (*c->p == ',') c->p++;
        jws(c);
    }
    if (*c->p == ']') c->p++;
}

static void load_added(impl_t *t, jc *c, load_ctx *lc) {
    jws(c);
    if (!jtake(c, '[')) { lc->failed = 1; tok_error(lc->error, lc->es, "bad added_tokens"); return; }
    jws(c);
    while (*c->p && *c->p != ']') {
        jws(c);
        if (!jtake(c, '{')) { lc->failed = 1; return; }
        char *content = NULL;
        uint32_t id = 0;
        jws(c);
        while (*c->p && *c->p != '}') {
            char *key = jstring(c, lc->error, lc->es);
            if (!key) { lc->failed = 1; return; }
            jws(c);
            if (!jtake(c, ':')) { free(key); lc->failed = 1; return; }
            jws(c);
            if (strcmp(key, "content") == 0) {
                content = jstring(c, lc->error, lc->es);
            } else if (strcmp(key, "id") == 0) {
                id = (uint32_t)strtoul(c->p, (char**)&c->p, 10);
                jskip(c);
            } else {
                jskip(c);
            }
            free(key);
            jws(c);
            if (*c->p == ',') c->p++;
            jws(c);
        }
        if (*c->p == '}') c->p++;
        if (content) {
            map_put(t->added_tokens, content, strlen(content), id);
            if (id > t->max_id) t->max_id = id;
            free(content);
        }
        jws(c);
        if (*c->p == ',') c->p++;
        jws(c);
    }
    if (*c->p == ']') c->p++;
}

static h3_tokenizer *load_tokenizer(const char *path, char *error, size_t error_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { tok_error(error, error_size, "cannot read tokenizer"); return NULL; }
    long sz;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); tok_error(error, error_size, "oom"); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';

    impl_t *t = calloc(1, sizeof(*t));
    if (!t) { free(buf); tok_error(error, error_size, "oom"); return NULL; }
    t->vocab = map_new(1 << 18);
    t->merge_ranks = map_new(1 << 18);
    t->added_tokens = map_new(128);
    t->bpe_cache = map_new(256);
    if (!t->vocab || !t->merge_ranks || !t->added_tokens || !t->bpe_cache) {
        free(buf); free(t); tok_error(error, error_size, "oom"); return NULL;
    }

    /* byte encoder/decoder */
    t->byte_encoder = calloc(256, sizeof(char*));
    for (int i = 0; i < 324; i++) t->byte_decoder[i] = -1;
    unsigned extra = 0;
    for (unsigned byte = 0; byte < 256; byte++) {
        int visible = (byte >= '!' && byte <= '~') ||
                      (byte >= 0xa1 && byte <= 0xac) ||
                      (byte >= 0xae && byte <= 0xff);
        uint32_t cp = visible ? byte : 256 + extra++;
        char tmp[8];
        utf16_to_utf8((UChar[]){ (UChar)cp }, 1, tmp, sizeof(tmp));
        t->byte_encoder[byte] = strdup(tmp);
        if (cp < 324) t->byte_decoder[cp] = (int16_t)byte;
    }

    jc cur = { buf };
    load_ctx lc = { t, error, error_size, 0 };
    /* walk top-level object */
    jws(&cur);
    if (!jtake(&cur, '{')) { lc.failed = 1; tok_error(error, error_size, "expected object"); }
    while (!lc.failed && *cur.p && *cur.p != '}') {
        char *key = jstring(&cur, error, error_size);
        if (!key) { lc.failed = 1; break; }
        jws(&cur);
        if (!jtake(&cur, ':')) { free(key); lc.failed = 1; break; }
        if (strcmp(key, "model") == 0) {
            /* object with vocab + merges */
            jws(&cur);
            if (!jtake(&cur, '{')) { free(key); lc.failed = 1; break; }
            jws(&cur);
            while (!lc.failed && *cur.p && *cur.p != '}') {
                char *mkey = jstring(&cur, error, error_size);
                if (!mkey) { lc.failed = 1; break; }
                jws(&cur);
                if (!jtake(&cur, ':')) { free(mkey); lc.failed = 1; break; }
                if (strcmp(mkey, "vocab") == 0) load_vocab(t, &cur, &lc);
                else if (strcmp(mkey, "merges") == 0) load_merges(t, &cur, &lc);
                else jskip(&cur);
                free(mkey);
                jws(&cur);
                if (*cur.p == ',') cur.p++;
                jws(&cur);
            }
            if (*cur.p == '}') cur.p++;
        } else if (strcmp(key, "added_tokens") == 0) {
            load_added(t, &cur, &lc);
        } else {
            jskip(&cur);
        }
        free(key);
        jws(&cur);
        if (*cur.p == ',') cur.p++;
        jws(&cur);
    }
    free(buf);
    if (lc.failed) {
        h3_tokenizer_free((h3_tokenizer*)t);
        return NULL;
    }

    /* inverse vocab */
    t->inverse_vocab = calloc(t->max_id + 1, sizeof(char*));
    /* build from vocab map */
    for (size_t i = 0; i < t->vocab->size; i++) {
        for (tok_map_entry *e = t->vocab->slots[i]; e; e = e->next) {
            if (e->value <= t->max_id) t->inverse_vocab[e->value] = e->key;
        }
    }
    /* inverse added */
    t->inverse_added = calloc(t->max_id + 1, sizeof(char*));
    for (size_t i = 0; i < t->added_tokens->size; i++) {
        for (tok_map_entry *e = t->added_tokens->slots[i]; e; e = e->next) {
            if (e->value <= t->max_id) t->inverse_added[e->value] = e->key;
        }
    }
    /* added alternatives sorted: longest first, then lexicographic */
    t->added_count = 0;
    for (size_t i = 0; i < t->added_tokens->size; i++)
        for (tok_map_entry *e = t->added_tokens->slots[i]; e; e = e->next) t->added_count++;
    if (t->added_count) {
        t->added_alternatives = malloc(t->added_count * sizeof(char*));
        size_t n = 0;
        for (size_t i = 0; i < t->added_tokens->size; i++)
            for (tok_map_entry *e = t->added_tokens->slots[i]; e; e = e->next)
                t->added_alternatives[n++] = strdup(e->key);
        /* sort: longer first, tie lexicographic */
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++) {
                size_t li = strlen(t->added_alternatives[i]);
                size_t lj = strlen(t->added_alternatives[j]);
                if (li < lj || (li == lj && strcmp(t->added_alternatives[i], t->added_alternatives[j]) > 0)) {
                    char *tmp = t->added_alternatives[i];
                    t->added_alternatives[i] = t->added_alternatives[j];
                    t->added_alternatives[j] = tmp;
                }
            }
    }
    return (h3_tokenizer*)t;
}

/* ---- decode ---------------------------------------------------------------- */
static char *tokenize_decode(impl_t *t, const uint32_t *ids, size_t count,
                             char *error, size_t error_size) {
    char *result = NULL;
    size_t rn = 0, rcap = 0;
    /* byte accumulator */
    char *bytes = NULL;
    size_t bn = 0, bcap = 0;
    /* append a byte to accumulator */
    /* flush bytes as UTF-8 */
    /* We simply append raw bytes to result; byte-level tokens are exact UTF-8 */
    for (size_t i = 0; i < count; i++) {
        uint32_t id = ids[i];
        if (id > t->max_id) {
            tok_error(error, error_size, "token ID is out of range");
            free(result); free(bytes);
            return NULL;
        }
        char *added = t->inverse_added[id];
        if (added) {
            /* flush bytes then append added content */
            size_t al = strlen(added);
            if (rn + al > rcap) { rcap = rn + al; result = realloc(result, rcap + 1); }
            memcpy(result + rn, added, al); rn += al;
            continue;
        }
        char *symbol = t->inverse_vocab[id];
        if (!symbol) {
            tok_error(error, error_size, "unknown token ID");
            free(result); free(bytes);
            return NULL;
        }
        /* byte-decode symbol */
        UChar *s = malloc(strlen(symbol) * 2 + 2);
        size_t slen = utf8_to_utf16(symbol, s, strlen(symbol) * 2 + 2);
        for (size_t j = 0; j < slen; j++) {
            uint32_t cp = s[j];
            if (cp >= 0xD800 && cp <= 0xDBFF && j + 1 < slen && s[j+1] >= 0xDC00 && s[j+1] <= 0xDFFF) {
                cp = 0x10000 + ((uint32_t)(s[j] - 0xD800) << 10) + (s[j+1] - 0xDC00);
                j++;
            }
            if (cp >= 324 || t->byte_decoder[cp] < 0) {
                free(s); free(result); free(bytes);
                tok_error(error, error_size, "invalid byte-level token");
                return NULL;
            }
            unsigned char byte = (unsigned char)t->byte_decoder[cp];
            if (bn + 1 > bcap) { bcap = bn + 1; bytes = realloc(bytes, bcap); }
            bytes[bn++] = (char)byte;
        }
        free(s);
    }
    /* append final bytes */
    if (bn) {
        if (rn + bn > rcap) { rcap = rn + bn; result = realloc(result, rcap + 1); }
        memcpy(result + rn, bytes, bn); rn += bn;
    }
    free(bytes);
    if (!result) result = strdup("");
    result[rn] = '\0';
    return result;
}

/* ======================= public API ======================= */

h3_tokenizer *h3_tokenizer_load(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) { tok_error(error, error_size, "tokenizer path is required"); return NULL; }
    return load_tokenizer(path, error, error_size);
}

void h3_tokenizer_free(h3_tokenizer *opaque) {
    impl_t *t = (impl_t*)opaque;
    if (!t) return;
    map_free(t->vocab);
    map_free(t->merge_ranks);
    map_free(t->added_tokens);
    map_free(t->bpe_cache);
    free(t->inverse_vocab);
    free(t->inverse_added);
    if (t->added_alternatives)
        for (size_t i = 0; i < t->added_count; i++) free(t->added_alternatives[i]);
    free(t->added_alternatives);
    if (t->byte_encoder)
        for (int i = 0; i < 256; i++) free(t->byte_encoder[i]);
    free(t->byte_encoder);
    free(t->cache_ids);
    free(t->cache_counts);
    free(t);
}

int h3_tokenizer_encode(const h3_tokenizer *opaque, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || !utf8 || !ids || !count) return 0;
    *ids = NULL;
    *count = 0;
    impl_t *t = (impl_t*)opaque;

    UChar *buf = malloc((strlen(utf8) + 1) * 2 + 2);
    if (!buf) { tok_error(error, error_size, "oom"); return 0; }
    size_t ulen = utf8_to_utf16(utf8, buf, (strlen(utf8) + 1) * 2 + 2);

    /* NFC normalize */
    UErrorCode ec = 0;
    const UNormalizer2 *nfc = unorm2_getNFCInstance_74(&ec);
    UChar *norm = malloc((ulen * 3 + 2) * sizeof(UChar));
    size_t nlen = (size_t)unorm2_normalize_74(nfc, buf, (int32_t)ulen, norm,
                                              (int32_t)(ulen * 3 + 2), &ec);
    free(buf);
    if (ec < 0) { free(norm); tok_error(error, error_size, "NFC normalization failed"); return 0; }

    uint32_t *out = NULL;
    size_t nout = 0, cap = 0;
    size_t start = 0;
    int ok = 1;
    while (start < nlen) {
        size_t mloc, mlen;
        uint32_t tokid = 0;
        if (!added_match(t, norm, nlen, start, &mloc, &mlen, &tokid)) break;
        if (mloc > start) {
            /* encode plain segment [start, mloc) */
            char *seg = malloc((mloc - start) * 3 + 1);
            utf16_to_utf8(norm + start, mloc - start, seg, (mloc - start) * 3 + 1);
            uint32_t *pids; size_t pcnt;
            if (!encode_plain(t, seg, &pids, &pcnt, error, error_size)) {
                free(seg); free(norm); free(out);
                return 0;
            }
            free(seg);
            if (nout + pcnt > cap) { cap = (nout + pcnt) * 2; out = realloc(out, cap * sizeof(uint32_t)); }
            memcpy(out + nout, pids, pcnt * sizeof(uint32_t));
            nout += pcnt;
            free(pids);
        }
        if (nout + 1 > cap) { cap = nout + 2; out = realloc(out, cap * sizeof(uint32_t)); }
        out[nout++] = tokid;
        start = mloc + mlen;
    }
    if (ok && start < nlen) {
        char *seg = malloc((nlen - start) * 3 + 1);
        utf16_to_utf8(norm + start, nlen - start, seg, (nlen - start) * 3 + 1);
        uint32_t *pids; size_t pcnt;
        if (!encode_plain(t, seg, &pids, &pcnt, error, error_size)) {
            free(seg); free(norm); free(out);
            return 0;
        }
        free(seg);
        if (nout + pcnt > cap) { cap = (nout + pcnt) * 2; out = realloc(out, cap * sizeof(uint32_t)); }
        memcpy(out + nout, pids, pcnt * sizeof(uint32_t));
        nout += pcnt;
        free(pids);
    }
    free(norm);
    if (nout == 0 && pad_empty) {
        if (nout + 1 > cap) { cap = 1; out = realloc(out, cap * sizeof(uint32_t)); }
        out[nout++] = H3_PAD_TOKEN_ID;
    }
    if (nout) *ids = out;
    *count = nout;
    return 1;
}

char *h3_tokenizer_decode(const h3_tokenizer *opaque,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || (!ids && count)) return NULL;
    return tokenize_decode((impl_t*)opaque, ids, count, error, error_size);
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}
