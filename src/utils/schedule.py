# Copyright 2020 fast.ai

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# One cycle policy LR calculation is copied from fast.ai v3 course:
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/05_anneal.ipynb
import math
import torch
from functools import partial


def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner


@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + list(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        if idx == 2: idx = 1
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


class OneCyclePolicy(object):
    """
    External interface made partially compatible with `timm.scheduler.scheduler.Scheduler`
    to avoid editing code to often, and the idea of stateless schedulers is very nice too.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_steps=None,
                 pct_warm=0.3,
                 max_lr=1e-3,
                 base_mom=0.85,
                 max_mom=0.95,
                 start_div_factor=25.0,
                 end_div_factor=10000.0):
        self.optimizer = optimizer
        self.cycle_steps = cycle_steps
        self.pct_warm = pct_warm
        self.max_lr = max_lr
        self.start_div_factor = start_div_factor
        self.end_div_factor = end_div_factor
        self.base_mom = base_mom
        self.max_mom = max_mom

        phases = [pct_warm, 1. - pct_warm]
        self.lr_schedule = combine_scheds(phases,
                                          [sched_cos(max_lr / start_div_factor, max_lr),
                                           sched_cos(max_lr, max_lr / end_div_factor)])

        self.mom_schedule = combine_scheds(phases,
                                           [sched_lin(max_mom, base_mom),
                                            sched_lin(base_mom, max_mom)])

        if "momentum" in optimizer.defaults:
            self.mom_field = "momentum"
        elif "betas" in optimizer.defaults:
            self.mom_field = "betas.0"
        else:
            self.mom_field = None

    def cur_pos(self, step: int):
        cur_step = step - (step // self.cycle_steps) * self.cycle_steps
        return cur_step / self.cycle_steps

    def step(self, epoch: int) -> None:
        pass

    def step_update(self, step: int) -> None:
        t = self.cur_pos(step)
        lr = self.lr_schedule(t)
        mom = self.mom_schedule(t)
        self.update_params(lr, "lr")
        self.update_params(mom, self.mom_field)

    def update_params(self, values, field: str):
        param_groups = self.optimizer.param_groups
        if not isinstance(values, (list, tuple)):
            values = [values] * len(param_groups)

        fields = field.split(".")
        if len(fields) > 1:
            field, idx = fields
            idx = int(idx)

            def _upd(group, val):
                tup = list(group[field])
                tup[idx] = val
                group[field] = tuple(tup)
        else:
            def _upd(group, val):
                group[field] = val

        for param_group, value in zip(param_groups, values):
            _upd(param_group, value)
