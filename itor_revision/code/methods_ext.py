"""Extended completion-aware methods for the EAAI track.

Every method starts from the same verified initial layout, keeps the best
verified complete layout as the output incumbent, and admits an objective
value into the anytime record only after the independent verifier of the
archive has accepted it.  The archive implementations of M0 and M1 are called
unchanged; the new methods only decide, before each reconstruction, which
reconstruction mode to run.

Method tokens
-------------
M0, M1            archive methods (completion_search.run_anytime)
CAP:<c>           M1 with repair cap c (CAP:0 is the reuse-only control R0)
SW:fixed<tau>     M1 (cap 4) until tau seconds after initialization, then M0
SW:stag<k>        M1 (cap 4); after k consecutive reconstructions without an
                  incumbent gain, switch permanently to M0
CAPSCHED:<k>      repair cap 4 -> 8 -> 16 -> M0, advancing one level after k
                  consecutive stagnant reconstructions
INTERLEAVE        alternate M1 (cap 4) and M0 reconstructions
BANDIT:<w>        sliding-window UCB over arms {cap 4, cap 8, cap 16, M0};
                  reward is incumbent gain per second, normalized online
"""

import math
import re
import time

import numpy as np

from completion_search import METHODS, _empty_stats, _reconstruct, _record, run_anytime
from layout_verifier import verify_complete_layout
from repair_baseline import CONFIGURATIONS, beam_repair, destroy, regret_repair

BASE_CAP = 4
BANDIT_ARMS = (("M1", 4), ("M1", 8), ("M1", 16), ("M0", None))
CAPSCHED_LEVELS = (("M1", 4), ("M1", 8), ("M1", 16), ("M0", None))
# Portfolio bandits add an incumbent-based destroy-and-repair round (the ALNS
# operators of the archive with adaptive weights kept across rounds) as an arm.
BANDITP_ARMS = (("M1", 4), ("M1", 8), ("M1", 16), ("M0", None), ("ALNS", None))
BANDITP3_ARMS = (("M1", 4), ("M0", None), ("ALNS", None))
ALNS_ROUND_FLOOR_S = 0.05
ALNS_ARM_CONFIGURATION = "small_beam"

_PATTERNS = [
    ("CAP", re.compile(r"^CAP:(\d+)$")),
    ("SW_FIXED", re.compile(r"^SW:fixed(\d+(?:\.\d+)?)$")),
    ("SW_STAG", re.compile(r"^SW:stag(\d+)$")),
    ("CAPSCHED", re.compile(r"^CAPSCHED:(\d+)$")),
    ("INTERLEAVE", re.compile(r"^INTERLEAVE$")),
    ("BANDIT", re.compile(r"^BANDIT:(\d+)$")),
    ("BANDIT2", re.compile(r"^BANDIT2:(\d+)$")),
    ("BANDITP", re.compile(r"^BANDITP:(\d+)$")),
    ("BANDITP3", re.compile(r"^BANDITP3:(\d+)$")),
    # Nonadaptive controls over the PORT3 arm set. They isolate the feedback
    # rule from the mere availability of the three operators.
    ("RR3", re.compile(r"^RR3$")),
    ("RND3", re.compile(r"^RND3$")),
    ("LRN", re.compile(r"^LRN:([A-Za-z0-9_]+)$")),
    ("SWL", re.compile(r"^SWL:([A-Za-z0-9_]+)$")),
    ("BANDITW", re.compile(r"^BANDITW:([A-Za-z0-9_]+)$")),
]
BANDIT2_ARMS = (("M1", 4), ("M0", None))
BANDITW_WINDOW = 8
BANDITW_PRIOR_WEIGHT = 2


def parse_method(token):
    if token in METHODS:
        return {"kind": "ARCHIVE", "method": token}
    for kind, pattern in _PATTERNS:
        match = pattern.match(token)
        if match:
            if kind in ("LRN", "SWL", "BANDITW"):
                return {"kind": kind, "value": match.group(1)}
            value = float(match.group(1)) if match.groups() else None
            return {"kind": kind, "value": value}
    raise ValueError("unknown method token: " + token)


def is_extended(token):
    return parse_method(token)["kind"] != "ARCHIVE"


class _Policy:
    """Chooses (mode, cap) for the next reconstruction and learns from its outcome."""

    def __init__(self, spec, start, rng):
        self.kind = spec["kind"]
        self.value = spec["value"]
        self.start = start
        self.rng = rng
        self.stagnant = 0
        self.level = 0
        self.switched = False
        self.bandit = self.kind in ("BANDIT", "BANDIT2", "BANDITP", "BANDITP3")
        self.window = int(self.value) if self.bandit else 0
        self.arms = {"BANDIT2": BANDIT2_ARMS, "BANDITP": BANDITP_ARMS,
                     "BANDITP3": BANDITP3_ARMS, "RR3": BANDITP3_ARMS,
                     "RND3": BANDITP3_ARMS}.get(self.kind, BANDIT_ARMS)
        self.history = []          # (arm index, normalized reward)
        self.reward_scale = 1e-9   # running maximum of raw rewards
        self.current_arm = 0
        self.durations = []        # durations of non-ALNS rounds, for ALNS round length matching
        # Warm start: learned prior rewards enter the window as pseudo-observations
        # and are displaced by real observations as the window slides.
        prior = spec.get("prior")
        if prior is not None:
            for arm, value in enumerate(list(prior)):
                for _ in range(BANDITW_PRIOR_WEIGHT):
                    self.history.append((arm, float(value)))

    def choose(self, iteration, now):
        if self.kind == "SW_FIXED":
            if now - self.start >= self.value:
                self.switched = True
            return ("M0", None) if self.switched else ("M1", BASE_CAP)
        if self.kind == "SW_STAG":
            if self.stagnant >= int(self.value):
                self.switched = True
            return ("M0", None) if self.switched else ("M1", BASE_CAP)
        if self.kind == "CAPSCHED":
            if self.stagnant >= int(self.value) and self.level < len(CAPSCHED_LEVELS) - 1:
                self.level += 1
                self.stagnant = 0
            return CAPSCHED_LEVELS[self.level]
        if self.kind == "INTERLEAVE":
            return ("M1", BASE_CAP) if iteration % 2 == 0 else ("M0", None)
        if self.kind == "RR3":
            # Round robin over the PORT3 arms: same operators, no feedback.
            return self.arms[iteration % len(self.arms)]
        if self.kind == "RND3":
            # Uniform random over the PORT3 arms: same operators, no feedback.
            return self.arms[int(self.rng.integers(len(self.arms)))]
        if self.bandit:
            recent = self.history[-self.window:] if self.window else self.history
            counts = np.zeros(len(self.arms))
            sums = np.zeros(len(self.arms))
            for arm, reward in recent:
                counts[arm] += 1
                sums[arm] += reward
            unplayed = [k for k in range(len(self.arms)) if counts[k] == 0]
            if unplayed:
                self.current_arm = unplayed[0]
            else:
                total = counts.sum()
                ucb = sums / counts + np.sqrt(2.0 * math.log(total) / counts)
                self.current_arm = int(np.argmax(ucb))
            return self.arms[self.current_arm]
        raise ValueError(self.kind)

    def update(self, mode, gain, elapsed, initial_value):
        improved = gain > 1e-12
        self.stagnant = 0 if improved else self.stagnant + 1
        if mode != "ALNS":
            self.durations.append(elapsed)
        if self.bandit:
            raw = 100.0 * gain / max(initial_value, 1e-9) / max(elapsed, 1e-6)
            self.reward_scale = max(self.reward_scale, raw)
            self.history.append((self.current_arm, raw / self.reward_scale))

    def alns_round_seconds(self):
        recent = self.durations[-3:]
        return max(ALNS_ROUND_FLOOR_S, float(np.median(recent))) if recent else ALNS_ROUND_FLOOR_S


class _ALNSState:
    """Adaptive destroy-and-repair state that persists across bandit rounds."""

    def __init__(self, problem, initial_value, configuration=ALNS_ARM_CONFIGURATION):
        self.config = dict(CONFIGURATIONS[configuration])
        self.operators = [(a, b) for a in ("spatial", "random", "low_contribution")
                          for b in ("beam", "regret")]
        self.weights = np.ones(len(self.operators))
        scale = max(abs(initial_value - problem.inst.weights.no_overlap_bonus), 1.0)
        self.temperature = self.config["temperature_fraction"] * scale
        self.iterations = 0


def _alns_round(problem, incumbent, incumbent_value, rng, start, deadline, events, stats,
                state, round_seconds):
    """Destroy-and-repair iterations from the incumbent for about round_seconds.

    Mirrors repair_baseline.run_adaptive_repair: roulette operator choice,
    beam or regret reinsertion with the original position as a candidate,
    temperature acceptance for the working layout, weight update from the
    outcome.  Every complete candidate goes through the common verifier and
    is admitted to the anytime record only if it finishes before the deadline.
    """
    current, current_value = list(incumbent), incumbent_value
    best_seen = incumbent_value
    round_start = time.perf_counter()
    done = 0
    while time.perf_counter() < deadline and (done == 0 or time.perf_counter() - round_start < round_seconds):
        index = int(rng.choice(len(state.operators), p=state.weights / state.weights.sum()))
        removal, insertion = state.operators[index]
        kept, removed, originals = destroy(problem, current, removal, state.config["destroy_cap"], rng)
        stats["destroy_sizes"].append(len(removed))
        repair = beam_repair if insertion == "beam" else regret_repair
        candidate = repair(problem, kept, removed, originals, rng, state.config, deadline)
        reward = 0.0
        if candidate is not None and _record(problem, candidate, start, deadline, events, stats, "adaptive-repair"):
            value = events[-1][1]
            delta = value - current_value
            accepted = delta > 1e-12 or rng.random() < math.exp(min(0.0, delta / max(state.temperature, 1e-12)))
            if accepted:
                reward = 3.0 if delta > 1e-12 else 1.0
                current, current_value = candidate, value
            if value > best_seen + 1e-12:
                reward = 6.0
                best_seen = value
        state.weights[index] = max(0.2, 0.9 * state.weights[index] + 0.1 * reward)
        state.temperature *= 0.999
        state.iterations += 1
        done += 1
    return done


def run_anytime_ext(problem, initial_witness, method, budgets, seed=0, kappa=32,
                    completion_tries=1, initial_value=None):
    """Run one method to the largest budget and sample its anytime trace.

    The output format matches completion_search.run_anytime, with an added
    stats["schedule"] list describing every reconstruction: mode, cap, start
    time, duration, incumbent gain and verified events.
    """
    spec = parse_method(method)
    if spec["kind"] == "ARCHIVE":
        return run_anytime(problem, initial_witness, method, budgets, seed=seed, kappa=kappa,
                           repair_cap=BASE_CAP, lns_cap=16, completion_tries=completion_tries,
                           initial_value=initial_value)
    if spec["kind"] == "LRN":
        import learned_methods
        return learned_methods.run_learned_anytime(problem, initial_witness, spec["value"], budgets,
                                                   seed=seed, initial_value=initial_value)
    if spec["kind"] == "CAP":
        cap = int(spec["value"])
        output = run_anytime(problem, initial_witness, "R0" if cap == 0 else "M1", budgets,
                             seed=seed, kappa=kappa, repair_cap=max(cap, 1), lns_cap=16,
                             completion_tries=completion_tries, initial_value=initial_value)
        output["method"] = method
        return output

    budgets = sorted(float(b) for b in budgets)
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets must be nonnegative")
    rng = np.random.default_rng(seed)
    stats = _empty_stats()
    stats["schedule"] = []
    if initial_value is None:
        initial_value = problem.score(initial_witness)
    if spec["kind"] == "SWL":
        # Learned switch time: search-free instance features -> tau, then SW:fixed<tau>.
        import learned_switch
        bundle = learned_switch.load_bundle(spec["value"])
        features = learned_switch.instance_features(problem, initial_value)
        tau = learned_switch.predict_tau(bundle, features)
        stats["predicted_tau_s"] = tau
        stats["switch_model"] = spec["value"]
        spec = {"kind": "SW_FIXED", "value": tau}
    elif spec["kind"] == "BANDITW":
        # Bandit warm-started with learned per-arm prior rewards from instance features.
        import learned_switch
        bundle = learned_switch.load_bundle(spec["value"])
        features = learned_switch.instance_features(problem, initial_value)
        prior = learned_switch.predict_arm_priors(bundle, features)
        stats["prior_rewards"] = [float(p) for p in prior]
        stats["prior_model"] = spec["value"]
        spec = {"kind": "BANDIT", "value": float(BANDITW_WINDOW), "prior": prior}
    start = time.perf_counter()
    deadline = start + budgets[-1]
    events = [(0.0, initial_value, list(initial_witness), "initial")]
    incumbent = list(initial_witness)
    best_value = initial_value
    iteration = 0
    policy = _Policy(spec, start, rng)
    alns_state = None
    alns_iterations = 0

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        mode, cap = policy.choose(iteration, now)
        before = len(events)
        before_best = best_value
        if mode == "ALNS":
            if alns_state is None:
                alns_state = _ALNSState(problem, initial_value)
            alns_iterations += _alns_round(problem, incumbent, best_value, rng, start, deadline, events,
                                           stats, alns_state, policy.alns_round_seconds())
        else:
            _reconstruct(problem, incumbent, mode, rng, start, deadline, events, stats, kappa,
                         cap if cap is not None else BASE_CAP, completion_tries,
                         random_order=iteration > 0)
        for _, value, layout, branch in events[before:]:
            if value > best_value + 1e-12:
                gain = value - best_value
                best_value = value
                incumbent = list(layout)
                counts = stats["incumbent_improvements_by_branch"]
                counts[branch] = counts.get(branch, 0) + 1
                gains = stats["incumbent_gain_by_branch"]
                gains[branch] = gains.get(branch, 0.0) + gain
        finished_iteration = time.perf_counter()
        gain = best_value - before_best
        stats["schedule"].append({"iteration": iteration, "mode": mode, "cap": cap,
                                  "start_s": now - start,
                                  "elapsed_s": finished_iteration - now,
                                  "gain": gain, "events": len(events) - before})
        policy.update(mode, gain, finished_iteration - now, initial_value)
        iteration += 1

    finished = time.perf_counter()
    stats["actual_search_elapsed_s"] = finished - start
    stats["overrun_s"] = max(0.0, finished - deadline)
    stats["max_admitted_event_s"] = max(event[0] for event in events)
    rows = []
    for budget in budgets:
        eligible = [(value, layout) for elapsed, value, layout, _ in events
                    if elapsed <= budget + 1e-12]
        value, layout = max(eligible, key=lambda item: item[0])
        check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, layout)
        if not check.valid:
            raise RuntimeError("anytime incumbent failed final verification")
        rows.append({"budget_s": budget, "J": value, "improvement": value - initial_value,
                     "improvement_pct": (100.0 * (value - initial_value) / initial_value
                                         if initial_value > 0 else None),
                     "layout": [[int(i), int(r), int(c)] for i, r, c in layout]})
    stats["events"] = len(events)
    stats["iterations"] = iteration
    stats["mean_displaced"] = (float(np.mean(stats["displaced_sizes"]))
                               if stats["displaced_sizes"] else None)
    stats["max_displaced"] = (max(stats["displaced_sizes"]) if stats["displaced_sizes"] else None)
    stats["mean_destroy_size"] = (float(np.mean(stats["destroy_sizes"])) if stats["destroy_sizes"] else None)
    stats["alns_iterations"] = alns_iterations
    if alns_state is not None:
        stats["alns_operator_weights"] = {a + "/" + b: float(w) for (a, b), w in zip(alns_state.operators, alns_state.weights)}
    modes = [s["mode"] for s in stats["schedule"]]
    stats["reconstructions_by_mode"] = {m: modes.count(m) for m in sorted(set(modes))}
    stats["time_by_mode"] = {m: float(sum(s["elapsed_s"] for s in stats["schedule"] if s["mode"] == m))
                             for m in sorted(set(modes))}
    stats["first_M0_start_s"] = next((s["start_s"] for s in stats["schedule"] if s["mode"] == "M0"), None)
    trace = [{"elapsed_s": float(elapsed), "J": float(value), "branch": branch}
             for elapsed, value, _, branch in events]
    return {"method": method, "initial_J": initial_value, "rows": rows, "trace": trace,
            "stats": stats}
