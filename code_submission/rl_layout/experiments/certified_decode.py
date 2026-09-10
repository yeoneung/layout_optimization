"""
Witness-preserving certified decoding, and the guarantee it actually supports.

An earlier version of this decoder tested the policy's top-k positions against a
certificate and, when none passed, committed the policy's argmax anyway.  That
fallback breaks the induction of the completion theorem: after an uncertified
commitment the state need not admit any completion, so the guarantee proved in
the paper did not apply to the algorithm the paper ran.  The decoder here removes
the gap in the two places it existed.

  * The certificate returns a *witness* -- an explicit completion of the
    remaining facilities -- not just a yes/no.  The witness's own next placement
    is always inserted into the candidate set, so candidate truncation can never
    be the reason no admissible action is found.

  * There is no uncertified fallback for an active instance.  If the initial
    state carries no witness the decoder abstains and returns NOT-CERTIFIED.
    Inactive batch slots are padded only because the vectorized environment
    requires one action per slot; those padded layouts are never returned as
    certified solutions.

That yields a *selective* guarantee: every layout this decoder returns is
feasible.  It does not yield a *coverage* guarantee -- the decoder abstains
whenever the witness generator cannot complete the empty plate -- and the two are
reported separately, because coverage is exactly the witness generator's own
success rate and has nothing to do with the policy.

The policy's role inside the certified region is therefore quality, not
feasibility: it chooses among certified placements, and the witness is only the
fallback that keeps the guarantee intact.
"""

import time

import numpy as np
import torch

NOT_CERTIFIED = "NOT-CERTIFIED"


class FirstFitWitness:
    """First-fit largest-first completion, returned as an explicit placement list.

    `certify(occ, remaining)` returns (ok, witness) where witness is a list of
    (facility, row, col) in the order first-fit would place them.  Soundness is
    by construction: the witness is a completion, so a state it accepts is
    completable.  It is not complete -- it rejects states a cleverer packer could
    finish -- which costs coverage and never correctness.
    """

    def __init__(self, iw, ih, GH, GW, gw, gh):
        self.iw, self.ih = iw, ih
        self.GH, self.GW, self.gw, self.gh = GH, GW, gw, gh
        self._rows = np.arange(GH)[:, None]
        self._cols = np.arange(GW)[None, :]

    def _legal(self, occ, i):
        ii = np.zeros((self.GH + 1, self.GW + 1), dtype=np.int64)
        np.cumsum(np.cumsum(occ, axis=0), axis=1, out=ii[1:, 1:])
        w, h = int(self.iw[i]), int(self.ih[i])
        r1 = np.minimum(self._rows + h, self.GH)
        c1 = np.minimum(self._cols + w, self.GW)
        S = (ii[r1, c1] - ii[self._rows, c1] - ii[r1, self._cols]
             + ii[self._rows, self._cols])
        inside = (self._cols + w <= self.gw) & (self._rows + h <= self.gh)
        return (S == 0) & inside

    def verify(self, occ, remaining, witness):
        """Independently verify that ``witness`` completes ``occ``.

        The checker deliberately does not call the witness generator.  It is the
        small trusted component behind the guarantee: generation may be
        heuristic or randomized, while acceptance is deterministic.
        """
        if occ.shape != (self.GH, self.GW):
            return False
        if np.any(occ < 0) or np.any(occ > 1):
            return False
        remaining = [int(i) for i in remaining]
        if witness is None or len(witness) != len(remaining):
            return False
        ids = [int(i) for i, _, _ in witness]
        if len(set(ids)) != len(ids) or set(ids) != set(remaining):
            return False

        work = occ.copy()
        for i, r, c in witness:
            i, r, c = int(i), int(r), int(c)
            if r < 0 or c < 0 or r + int(self.ih[i]) > self.gh:
                return False
            if c + int(self.iw[i]) > self.gw:
                return False
            if not self._legal(work, i)[r, c]:
                return False
            work[r:r + int(self.ih[i]), c:c + int(self.iw[i])] += 1
        return True

    def certify(self, occ, remaining, tries=1, rng=None):
        """Attempt `tries` completions; return the first that succeeds.

        Attempt 0 is deterministic first-fit.  Later attempts randomize which
        legal position each facility takes, which finds completions first-fit
        misses.  Since coverage of the certified decoder is exactly this
        procedure's success rate at the empty plate, `tries` is the knob that
        trades certificate cost against how often the decoder can answer at all
        --- and it changes nothing about soundness, because every attempt that
        succeeds returns an explicit completion.
        """
        rng = rng or np.random.default_rng(0)
        order = sorted(remaining, key=lambda j: -(self.iw[j] * self.ih[j]))
        for t in range(tries):
            work = occ.copy()
            witness = []
            ok = True
            for i in order:
                m = self._legal(work, i)
                idx = np.argwhere(m)
                if not len(idx):
                    ok = False
                    break
                k = 0 if t == 0 else int(rng.integers(len(idx)))
                r, c = int(idx[k][0]), int(idx[k][1])
                work[r:r + int(self.ih[i]), c:c + int(self.iw[i])] += 1
                witness.append((int(i), r, c))
            if ok:
                if not self.verify(occ, remaining, witness):
                    raise RuntimeError("witness generator returned an invalid completion")
                return True, witness
        return False, None


@torch.no_grad()
def run_certified_decode(model, nrm, env, dev, inst, n_cand=32, temperature=1.0,
                         deterministic=False, seed=0, init_tries=1, step_tries=1):
    """Certified decoding for a batch of instances.

    Returns per-instance `certified` flags alongside the layouts.  Instances that
    abstain carry `certified=False` and their layout must not be read as a
    solution; the caller reports coverage from this flag.
    """
    t0 = time.time()
    B, n = env.B, env.n
    obs, glob = env.reset()
    T = torch.as_tensor

    iw = np.rint(inst.w).astype(np.int64)
    ih = np.rint(inst.h).astype(np.int64)
    wits = [FirstFitWitness(iw[b], ih[b], env.GH, env.GW,
                            int(round(inst.gw[b])), int(round(inst.gh[b])))
            for b in range(B)]
    occ = [np.zeros((env.GH, env.GW), dtype=np.int32) for _ in range(B)]
    placed = np.zeros((B, n), dtype=bool)

    # the guarantee's precondition, evaluated once per instance
    witness = [None] * B
    certified = np.zeros(B, dtype=bool)
    rng = np.random.default_rng(seed)
    for b in range(B):
        ok, w = wits[b].certify(occ[b], list(range(n)), tries=init_tries, rng=rng)
        certified[b] = ok
        witness[b] = w

    n_witness_used = 0
    n_reject = 0

    for _ in range(n):
        tok, pooled, _, rcat = model.room_dist(T(nrm(obs), device=dev),
                                               T(glob, device=dev),
                                               T(~env.placed, device=dev))
        fr = env.forced_room()
        if fr is not None:
            room = T(fr, device=dev)
        elif deterministic:
            room = rcat.logits.argmax(-1)
        else:
            room = torch.distributions.Categorical(
                logits=rcat.logits / temperature).sample()
        room_np = room.cpu().numpy()
        P, legal, dead, fb = env.planes(room_np)
        logits = model.pos_logits(tok, pooled, room, T(P, device=dev),
                                  T(legal, device=dev), T(dead, device=dev))
        order = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        pos = np.empty(B, dtype=np.int64)

        for b in range(B):
            i = int(room_np[b])
            if not certified[b]:
                pos[b] = int(order[b, 0])          # abstained; finish arbitrarily
                placed[b, i] = True
                continue

            rem = [j for j in range(n) if not placed[b, j] and j != i]
            # the witness's own next placement for this facility, if it has one
            wit_pos = None
            for (j, r, c) in witness[b]:
                if j == i:
                    wit_pos = r * env.GW + c
                    break

            chosen = None
            tried = 0
            for k in range(min(n_cand, order.shape[1])):
                f = int(order[b, k])
                if not np.isfinite(logits[b, f].item()):
                    break
                tried += 1
                r, c = f // env.GW, f % env.GW
                if not wits[b]._legal(occ[b], i)[r, c]:
                    n_reject += 1
                    continue
                trial = occ[b].copy()
                trial[r:r + ih[b, i], c:c + iw[b, i]] += 1
                ok, w = wits[b].certify(trial, rem, tries=step_tries, rng=rng)
                if ok:
                    chosen = (f, trial, w)
                    break
                n_reject += 1

            if chosen is None and wit_pos is not None:
                # The theorem's insurance.  Do not rerun a sufficient (and
                # therefore incomplete) generator here: the remainder of the
                # already verified witness is itself the certificate.
                r, c = wit_pos // env.GW, wit_pos % env.GW
                trial = occ[b].copy()
                trial[r:r + ih[b, i], c:c + iw[b, i]] += 1
                w = [(j, wr, wc) for (j, wr, wc) in witness[b] if j != i]
                if not wits[b].verify(trial, rem, w):
                    raise RuntimeError(
                        f"carried witness invariant failed for batch item {b}, facility {i}")
                chosen = (wit_pos, trial, w)
                n_witness_used += 1

            if chosen is None:
                raise RuntimeError(
                    f"certified decoder lost its witness for batch item {b}, facility {i}")

            pos[b], occ[b], witness[b] = chosen
            placed[b, i] = True

        obs, glob, _, _ = env.step(room_np, pos)

    x, y, tot, c = env.result()
    return {"x": x, "y": y, "best": tot, "overlap_area": c["overlap_area"],
            "certified": certified.copy(), "coverage": float(certified.mean()),
            "witness_used": n_witness_used, "rejects": n_reject,
            "evals": n * B, "wall_time": time.time() - t0}
