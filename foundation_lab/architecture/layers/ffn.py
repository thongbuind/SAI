from SAI.ffn import SwiGLU
from foundation_lab.architecture.registry import FFNS

# FFN của model chính thức; biến thể FFN mới đăng ký thêm tên khác vào FFNS.
FFNS.register("swiglu")(SwiGLU)
