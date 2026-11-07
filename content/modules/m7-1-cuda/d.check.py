import torch
import torch.nn.functional as F
from labcheck import close, need, ensure


def check(ns, R):
    ext = need(ns, "ext", "derlenmiş modül")

    def eq():
        g = torch.Generator().manual_seed(0)
        x, b = torch.randn(37, 129, generator=g) * 3, torch.randn(129, generator=g)
        close(ext.bias_gelu(x, b), F.gelu(x + b, approximate="tanh"), atol=1e-5)
    R.test("bias_gelu = F.gelu(x + b, tanh)", eq)

    def checks():
        x, b = torch.randn(8, 4), torch.randn(4)
        bad = [("transpoze (bitişik değil)", lambda: ext.bias_gelu(torch.randn(4, 8).t(), b)),
               ("float64", lambda: ext.bias_gelu(x.double(), b.double())),
               ("boyut uyuşmazlığı", lambda: ext.bias_gelu(x, torch.randn(5)))]
        for what, fn in bad:
            try:
                fn()
            except RuntimeError:
                continue
            raise AssertionError(f"{what}: TORCH_CHECK hata fırlatmalıydı")
    R.test("TORCH_CHECK'ler yanlış girdiyi reddediyor", checks)

    def cuda():
        if not torch.cuda.is_available():
            return "CUDA yok: GPU kısmı atlandı (Colab'da GPU çalışma zamanıyla dene)"
        src = ns.get("CUDA_SRC")
        if not src:
            return "CUDA var ama CUDA_SRC boş (isteğe bağlı)"
        import hashlib
        from torch.utils.cpp_extension import load_inline
        m = load_inline(name="ptl_bgc_" + hashlib.md5(src.encode()).hexdigest()[:8],
                        cpp_sources="torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b);",
                        cuda_sources=src, functions=["bias_gelu_cuda"], verbose=False)
        x, b = torch.randn(300, 77, device="cuda"), torch.randn(77, device="cuda")
        close(m.bias_gelu_cuda(x, b).cpu(), F.gelu(x + b, approximate="tanh").cpu(), atol=1e-5)
        return "CUDA çekirdeği doğru"
    R.test("GPU (isteğe bağlı)", cuda)
