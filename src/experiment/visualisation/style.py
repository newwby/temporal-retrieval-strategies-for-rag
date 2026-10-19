SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name == "bottom":
            spine.set_color(BASELINE)
        else:
            spine.set_visible(False)
    ax.tick_params(colors=INK_MUTED, length=0)
