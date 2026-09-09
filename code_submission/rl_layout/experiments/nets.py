"""
Shared networks.

`RoomEncoder` is the same room-wise-embedding -> self-attention -> pooling stack
the paper uses, with two additions that the instance-distribution setting forces:

  * no learned per-room identity embedding.  legacy gave each room index its own
    parameter, which is fine on a single fixed instance but is exactly what
    prevents transfer -- room 7 means something different in every instance.
    Identity has to come from the features (size, affinity row, wall bonuses)
    instead, which is what makes zero-shot evaluation on an unseen brief possible.
  * a small vector of episode-level features (progress, gap to the incumbent,
    temperature) appended to every token, so the policy can condition on where it
    is in its own search rather than only on the geometry.
"""

import math

import torch
import torch.nn as nn


class RoomEncoder(nn.Module):
    def __init__(self, feat_dim, n_glob, d_model=128, n_heads=4, n_blocks=3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(feat_dim + n_glob, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(n_blocks)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ff = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                           nn.Linear(2 * d_model, d_model)) for _ in range(n_blocks)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, obs, glob):
        """obs: (B, n, feat).  glob: (B, n_glob) episode-level features."""
        g = glob[:, None, :].expand(-1, obs.shape[1], -1)
        z = self.embed(torch.cat([obs, g], dim=-1))
        for attn, ln1, ff, ln2 in zip(self.attn, self.ln1, self.ff, self.ln2):
            h = ln1(z)
            a, _ = attn(h, h, h, need_weights=False)
            z = z + a
            z = z + ff(ln2(z))
        z = self.ln_out(z)
        return z, torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


# operation ids for the improvement policy
OP_TRANSLATE, OP_JUMP, OP_SWAP, OP_SNAP = 0, 1, 2, 3
N_OPS = 4
OP_NAMES = ["translate", "jump", "swap", "snap"]


class ImprovePolicy(nn.Module):
    """Factorized action: room -> operation -> operation parameters.

    Log-probabilities of the three levels add.  The parameter level differs per
    operation, so all branches are computed and the executed one is gathered --
    that keeps `evaluate()` (needed for the PPO ratio) exact rather than
    approximate.
    """

    def __init__(self, feat_dim, n_glob, d_model=128, n_blocks=3, ops=(0, 1, 2, 3)):
        super().__init__()
        self.enc = RoomEncoder(feat_dim, n_glob, d_model, n_blocks=n_blocks)
        self.d = d_model
        self.ops = list(ops)
        self.register_buffer("op_mask", torch.full((N_OPS,), float("-inf")))
        self.op_mask[list(ops)] = 0.0

        self.critic = mlp(2 * d_model, d_model, 1)
        self.room_head = mlp(d_model, d_model, 1)                 # pointer over rooms
        ctx = 3 * d_model                                          # token + mean + max
        self.op_head = mlp(ctx, d_model, N_OPS)
        self.delta_head = mlp(ctx, d_model, 2)                     # translate
        self.jump_head = mlp(ctx, d_model, 2)                      # absolute target
        self.swap_q = mlp(ctx, d_model, d_model)                   # swap: pointer query
        self.swap_k = nn.Linear(d_model, d_model)
        self.log_std_delta = nn.Parameter(torch.zeros(2) - 0.5)
        self.log_std_jump = nn.Parameter(torch.zeros(2) - 1.0)

    def _trunk(self, obs, glob):
        tok, pooled = self.enc(obs, glob)
        return tok, pooled, self.critic(pooled).squeeze(-1)

    def _ctx(self, tok, pooled, room):
        sel = tok[torch.arange(tok.shape[0], device=tok.device), room]
        return torch.cat([sel, pooled], dim=-1)

    def _dists(self, tok, pooled, room):
        ctx = self._ctx(tok, pooled, room)
        op_logits = self.op_head(ctx) + self.op_mask
        mu_d = torch.tanh(self.delta_head(ctx))
        mu_j = self.jump_head(ctx)
        q = self.swap_q(ctx)
        swap_logits = torch.einsum("bd,bnd->bn", q, self.swap_k(tok)) / math.sqrt(self.d)
        # a room may not swap with itself
        swap_logits = swap_logits.scatter(
            1, room[:, None], torch.full_like(swap_logits[:, :1], float("-inf")))
        return op_logits, mu_d, mu_j, swap_logits

    @staticmethod
    def _branch_logp(op, ent_parts, logps):
        """Gather the log-prob / entropy of the branch that was executed."""
        lp = torch.zeros_like(logps[0])
        en = torch.zeros_like(ent_parts[0])
        for k in range(N_OPS):
            m = (op == k)
            if m.any():
                lp = torch.where(m, logps[k], lp)
                en = torch.where(m, ent_parts[k], en)
        return lp, en

    def act(self, obs, glob, deterministic=False):
        tok, pooled, value = self._trunk(obs, glob)
        room_logits = self.room_head(tok).squeeze(-1)
        rcat = torch.distributions.Categorical(logits=room_logits)
        room = room_logits.argmax(-1) if deterministic else rcat.sample()

        op_logits, mu_d, mu_j, swap_logits = self._dists(tok, pooled, room)
        ocat = torch.distributions.Categorical(logits=op_logits)
        op = op_logits.argmax(-1) if deterministic else ocat.sample()

        sd = self.log_std_delta.exp().expand_as(mu_d)
        sj = self.log_std_jump.exp().expand_as(mu_j)
        nd = torch.distributions.Normal(mu_d, sd)
        nj = torch.distributions.Normal(mu_j, sj)
        delta = mu_d if deterministic else nd.sample()
        jump = mu_j if deterministic else nj.sample()
        scat = torch.distributions.Categorical(logits=swap_logits)
        swap = swap_logits.argmax(-1) if deterministic else scat.sample()

        logps = [nd.log_prob(delta).sum(-1), nj.log_prob(jump).sum(-1),
                 scat.log_prob(swap), torch.zeros_like(value)]
        ents = [nd.entropy().sum(-1), nj.entropy().sum(-1),
                scat.entropy(), torch.zeros_like(value)]
        plp, pen = self._branch_logp(op, ents, logps)

        logp = rcat.log_prob(room) + ocat.log_prob(op) + plp
        ent = rcat.entropy() + ocat.entropy() + pen
        return {"room": room, "op": op, "delta": delta, "jump": jump, "swap": swap,
                "logp": logp, "ent": ent, "value": value}

    def evaluate(self, obs, glob, a):
        tok, pooled, value = self._trunk(obs, glob)
        room = a["room"]
        room_logits = self.room_head(tok).squeeze(-1)
        rcat = torch.distributions.Categorical(logits=room_logits)
        op_logits, mu_d, mu_j, swap_logits = self._dists(tok, pooled, room)
        ocat = torch.distributions.Categorical(logits=op_logits)
        sd = self.log_std_delta.exp().expand_as(mu_d)
        sj = self.log_std_jump.exp().expand_as(mu_j)
        nd = torch.distributions.Normal(mu_d, sd)
        nj = torch.distributions.Normal(mu_j, sj)
        scat = torch.distributions.Categorical(logits=swap_logits)
        logps = [nd.log_prob(a["delta"]).sum(-1), nj.log_prob(a["jump"]).sum(-1),
                 scat.log_prob(a["swap"]), torch.zeros_like(value)]
        ents = [nd.entropy().sum(-1), nj.entropy().sum(-1),
                scat.entropy(), torch.zeros_like(value)]
        plp, pen = self._branch_logp(a["op"], ents, logps)
        return (rcat.log_prob(room) + ocat.log_prob(a["op"]) + plp,
                rcat.entropy() + ocat.entropy() + pen, value)


class PositionCNN(nn.Module):
    """Scores every lattice position for the room currently being placed.

    The input planes are local (occupancy, legality, the affinity of the room
    being placed towards whoever owns each cell, the wall bonus it would earn);
    what makes a position good is not.  Dilated residual convolutions widen the
    receptive field to cover the whole floor plate in four layers, and the room's
    own embedding modulates every channel (FiLM), so one shared trunk scores
    positions differently depending on which room is being placed.
    """

    def __init__(self, in_planes, d_ctx, ch=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Conv2d(in_planes, ch, 3, padding=1)
        self.film = nn.Linear(d_ctx, 2 * ch)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.GroupNorm(8, ch), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=d, dilation=d),
                nn.GroupNorm(8, ch), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=d, dilation=d)))
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.GELU(), nn.Conv2d(ch, 1, 1))

    def forward(self, planes, ctx):
        z = self.stem(planes)
        g, b = self.film(ctx).chunk(2, dim=-1)
        z = z * (1 + g[:, :, None, None]) + b[:, :, None, None]
        for blk in self.blocks:
            z = z + blk(z)
        return self.out(z).squeeze(1)                 # (B, GH, GW)


class ConstructPolicy(nn.Module):
    """Places rooms one at a time on the lattice.

    Step t is factorized: pick which unplaced room to commit next (pointer over
    room tokens), then pick where to put it (CNN over the floor plate, masked to
    positions that provably overlap nothing).  Feasibility is therefore a
    property of the action set rather than something the reward has to teach --
    which is where the improvement formulation spends nearly all of its samples.
    """

    def __init__(self, feat_dim, n_glob, in_planes, d_model=128, n_blocks=3, ch=64):
        super().__init__()
        self.enc = RoomEncoder(feat_dim, n_glob, d_model, n_blocks=n_blocks)
        self.d = d_model
        self.critic = mlp(2 * d_model, d_model, 1)
        self.room_head = mlp(d_model, d_model, 1)
        self.pos_cnn = PositionCNN(in_planes, 3 * d_model, ch=ch)

    def _trunk(self, obs, glob):
        tok, pooled = self.enc(obs, glob)
        return tok, pooled, self.critic(pooled).squeeze(-1)

    def room_dist(self, obs, glob, room_mask):
        tok, pooled, value = self._trunk(obs, glob)
        rl = self.room_head(tok).squeeze(-1).masked_fill(~room_mask, float("-inf"))
        return tok, pooled, value, torch.distributions.Categorical(logits=rl)

    def pos_logits(self, tok, pooled, room, planes, legal, dead):
        b = torch.arange(tok.shape[0], device=tok.device)
        ctx = torch.cat([tok[b, room], pooled], dim=-1)
        lg = self.pos_cnn(planes, ctx).flatten(1)
        m = legal.flatten(1)
        # a room with no legal position at all falls back to the least-overlapping
        # cell, which the environment supplies as `dead_logits`
        lg = torch.where(dead[:, None], torch.zeros_like(lg), lg)
        return lg.masked_fill(~m & ~dead[:, None], float("-inf"))

    def act(self, obs, glob, room_mask, planes_fn, deterministic=False, temperature=1.0,
            forced_room=None):
        """`planes_fn(room_np)` returns (planes, legal, dead) for the chosen room;
        the environment can only build them once the room is known.

        `temperature` sweeps the policy from near-deterministic to near-uniform
        over the legal set, which is what traces out its quality/diversity front
        without retraining -- the counterpart of varying an annealing budget.

        `forced_room` hands the commit order to the environment instead.  That
        makes the order an exogenous random draw, exactly like an annealer's seed,
        and it is the cheapest source of genuine variety a constructive method
        has: the same well-trained placement rule applied in a different order
        gives a different, still-good layout.  When the room is not the policy's
        decision its log-probability is not part of the objective either.
        """
        tok, pooled, value, rcat = self.room_dist(obs, glob, room_mask)
        if forced_room is not None:
            room = forced_room
        elif deterministic:
            room = rcat.logits.argmax(-1)
        else:
            room = torch.distributions.Categorical(logits=rcat.logits / temperature).sample()
        planes, legal, dead = planes_fn(room.cpu().numpy())
        lg = self.pos_logits(tok, pooled, room, planes, legal, dead)
        pcat = torch.distributions.Categorical(logits=lg)
        pos = lg.argmax(-1) if deterministic else \
            torch.distributions.Categorical(logits=lg / temperature).sample()
        if forced_room is None:
            logp = rcat.log_prob(room) + pcat.log_prob(pos)
            ent = rcat.entropy() + pcat.entropy()
        else:
            logp, ent = pcat.log_prob(pos), pcat.entropy()
        return {"room": room, "pos": pos, "value": value, "logp": logp, "ent": ent,
                "planes": planes, "legal": legal, "dead": dead}

    def evaluate(self, obs, glob, room_mask, planes, legal, dead, room, pos,
                 forced=False):
        tok, pooled, value, rcat = self.room_dist(obs, glob, room_mask)
        lg = self.pos_logits(tok, pooled, room, planes, legal, dead)
        pcat = torch.distributions.Categorical(logits=lg)
        if forced:
            return pcat.log_prob(pos), pcat.entropy(), value
        return (rcat.log_prob(room) + pcat.log_prob(pos),
                rcat.entropy() + pcat.entropy(), value)
