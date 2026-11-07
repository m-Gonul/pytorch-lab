PyTorch'ta im2col `F.unfold(x, kernel_size, padding=, stride=)` → `(N, C·KH·KW, L)`; col2im `F.fold(cols, output_size, kernel_size, padding=, stride=)`.

- `conv2d_unfold(x, w, b, stride=1, pad=0)` — `unfold` + matmul + reshape; `F.conv2d` ile aynı çıktı **ve** aynı autograd gradyanları.
- `conv2d_input_grad(dout, w, x_shape, stride=1, pad=0)` → `dx`; autograd kullanmadan: `cols = W_matᵀ @ dout_mat`, sonra `F.fold`. Kontrol, autograd'ın verdiği `dx` ile karşılaştırır.
