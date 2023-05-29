import matplotlib.pyplot as plt
import numpy as np


def bar_plot(success_rate, Nifg):
    # 柱状图
    plt.figure(figsize=(8, 5))
    plt.rc("font", family="FangSong")
    plt.bar(Nifg, success_rate, width=5, align="edge")
    for x, y in zip(Nifg, success_rate):
        # ha: horizontal alignment（横向对齐方式）， va: vertical alignment（纵向对齐方式）
        # x+2.5, y+0.05 为标签的坐标
        plt.text(x + 2.5, y + 0.005, "%.2f" % y, ha="center", va="bottom")
    plt.title("Bar", fontsize=20)
    plt.xlabel("Nifg", fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    # x轴刻度
    plt.xticks([index + 2.5 for index in Nifg], Nifg)
    plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/bar_plot.png")
