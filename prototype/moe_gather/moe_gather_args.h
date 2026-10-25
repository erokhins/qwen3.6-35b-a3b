/* moe_gather_args.h — ONE argv parser, used by both arms.
 *
 * Why this is a separate shared header instead of the two copies it replaced: HANDOFF §7.1 recorded
 * that `arg_value` matched only `argv[i] == "--arm"` and read `argv[i+1]`, so `--arm=1` fell through
 * to the default `0`. Four "arm" invocations were all arm 0. The same twelve lines sat in
 * `host_probe.cpp` and in `moe_gather.hip`, so the defect had to be fixed twice — and a fix applied
 * to one arm only would leave the two arms judging different requests, which is the failure mode
 * `pitfalls/40-gates-and-admissions.md` §"A default in a gate driver and the default in the binary
 * under test must move in one commit" is about. One spelling, one place.
 *
 * The rules this parser enforces, all of them the anti-fail-open kind:
 *   * `--flag value` AND `--flag=value` are the same thing (the defect being fixed);
 *   * an unrecognised `--token` is an ERROR, not something to ignore — a typo'd flag used to be a
 *     silently-defaulted flag, and a run that silently used the default arm prints PASS;
 *   * a positional argument is an error: nothing here takes one, so one means a shell quoting slip;
 *   * a value-taking flag must actually receive a value, and that value must not itself start with
 *     `--` (`--arm --plan p` is a missing value, not `--arm` = "--plan");
 *   * numeric flags are parsed with strtol/strtoll and must consume the whole token, so `--arm=x`
 *     or `--arm=1x` is refused instead of becoming `atoi`-zero.
 */
#ifndef MOE_GATHER_ARGS_H
#define MOE_GATHER_ARGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;         /* "--plan" */
    int takes_value;          /* 0 = boolean switch, 1 = "--plan p" / "--plan=p" */
} MgArgSpec;

static int mg_arg_spec_has(const MgArgSpec *specs, const char *name, int *takes_value) {
    for (const MgArgSpec *s = specs; s && s->name; ++s)
        if (!strcmp(s->name, name)) { if (takes_value) *takes_value = s->takes_value; return 1; }
    return 0;
}

/* Reports the first offending token in `bad`, returns 0 when the whole command line is legal. */
static int mg_arg_validate(int argc, char **argv, const MgArgSpec *specs, char *bad, size_t badn) {
    if (bad && badn) bad[0] = '\0';
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (a[0] != '-' || a[1] != '-') {
            if (bad && badn) snprintf(bad, badn, "positional-argument '%s'", a);
            return i;
        }
        const char *eq = strchr(a + 2, '=');
        char key[64];
        const size_t kl = eq ? (size_t)(eq - a) : strlen(a);
        if (kl >= sizeof(key)) {
            if (bad && badn) snprintf(bad, badn, "flag-too-long '%s'", a);
            return i;
        }
        memcpy(key, a, kl);
        key[kl] = '\0';
        int tv = 0;
        if (!mg_arg_spec_has(specs, key, &tv)) {
            if (bad && badn) snprintf(bad, badn, "unknown-flag '%s'", key);
            return i;
        }
        if (!tv) {
            if (eq) {
                if (bad && badn) snprintf(bad, badn, "boolean-flag-given-a-value '%s'", key);
                return i;
            }
            continue;
        }
        if (!eq) {
            if (i + 1 >= argc) {
                if (bad && badn) snprintf(bad, badn, "missing-value-for '%s'", key);
                return i;
            }
            if (!strncmp(argv[i + 1], "--", 2)) {
                if (bad && badn) snprintf(bad, badn, "missing-value-for '%s'", key);
                return i;
            }
            ++i;                                   /* consumed as the value */
        }
    }
    return 0;
}

/* Lookup happens AFTER mg_arg_validate has refused anything unknown, so these two stay dumb: they
 * never guess a default for a flag that does not exist. */
static const char *mg_arg_value(int argc, char **argv, const char *flag, const char *dflt) {
    const size_t fl = strlen(flag);
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, flag)) return (i + 1 < argc) ? argv[i + 1] : dflt;
        if (!strncmp(a, flag, fl) && a[fl] == '=') return a + fl + 1;
    }
    return dflt;
}

static int mg_arg_has(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

/* Strict integer fetch: `ok` is set to 0 when the token exists but is not a whole integer. */
static long long mg_arg_ll(int argc, char **argv, const char *flag, long long dflt, int *ok) {
    const char *v = mg_arg_value(argc, argv, flag, 0);
    if (ok) *ok = 1;
    if (!v) return dflt;
    char *end = 0;
    const long long r = strtoll(v, &end, 10);
    if (!end || *end != '\0' || end == v) { if (ok) *ok = 0; return dflt; }
    return r;
}

#endif /* MOE_GATHER_ARGS_H */
