class Registry:
    """Bảng tra tên → implementation của một phòng ban.

    Mỗi phòng (attention kernel, FFN, họ mô hình, nguồn dữ liệu, recipe...) có một registry.
    Implementation mới chỉ cần ``@REGISTRY.register("ten")`` rồi chọn bằng tên trong config.
    Tên trùng hoặc tên không tồn tại đều báo lỗi ngay, không âm thầm thay thế.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self._items = {}

    def register(self, name: str):
        def decorator(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' đã được đăng ký")
            self._items[name] = obj
            return obj
        return decorator

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(
                f"{self.kind} '{name}' không tồn tại; các lựa chọn: {self.names()}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self):
        return sorted(self._items.items())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"
