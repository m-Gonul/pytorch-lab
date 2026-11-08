from labcheck import need, ensure


def check(ns, R_):
    def t():
        ensure(need(ns, "params")("lstm", 10, 20) == 4 * (200 + 400 + 40), "LSTM (PyTorch)")
        ensure(ns["params"]("gru", 10, 20, False) == 3 * (200 + 400 + 20), "GRU (tek bias)")
        ensure(ns["params"]("peephole", 10, 20) == ns["params"]("lstm", 10, 20) + 60, "peephole +3h")
        ensure(need(ns, "stack_params")("lstm", 1, [16, 16]) == ns["params"]("lstm", 1, 16) + ns["params"]("lstm", 16, 16), "yığın")
        rows = []
        for cell in ("rnn", "gru", "cifg", "lstm", "peephole"):
            rows.append([cell, ns["params"](cell, 256, 512), f"{ns['params'](cell, 256, 512) / ns['params']('lstm', 256, 512):.2f}"])
        R_.table("d = 256, h = 512, PyTorch sayımı", ["hücre", "parametre", "LSTM'e oran"], rows,
                 note="GRU ve CIFG LSTM'in 3/4'ü kadardır; aynı parametre bütçesinde GRU daha geniş olabilir. Jozefowicz ve ark. (2015) 10 000 mimarilik aramada GRU'yu çoğu görevde LSTM'e denk ya da üstün buldu, ama unutma bias'ı 1 olan LSTM'in farkı kapattığını gösterdi (x2).")
    R_.test("parametre sayıları", t)
