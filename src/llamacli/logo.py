from pyfiglet import Figlet
from rich.console import Console


def get_logo_text():
    try:
        fig = Figlet(font="slant")
        return fig.renderText("llamacli")
    except Exception:
        fig = Figlet(font="standard")
        return fig.renderText("llamacli")


def print_logo(console: Console):
    logo = get_logo_text()
    console.print(logo, style="bold white")
    console.print("  Fine-tune LLMs in your terminal\n", style="dim")
