def amdahl(f, n):
    ...


def gustafson(f, n):
    ...


def karp_flatt(S, p):
    ...


def pipeline_makespan(stages, n_items):
    ...


def work_span(dur, preds):
    ...


def list_schedule(dur, preds, p):
    ...


dur = {"a": 2, "b": 3, "c": 1, "d": 4}
preds = {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]}
print(work_span(dur, preds), list_schedule(dur, preds, 2))
