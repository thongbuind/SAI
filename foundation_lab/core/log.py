LINE = "╠════════════════════════════════════════════════════════════════════════════════════╣"

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

def banner(title: str):
    """In tiêu đề một khối log theo khung."""
    print(LINE)
    print(f"║{title.center(84)}║")
    print(LINE)
