import matplotlib.pyplot as plt
import numpy as np
import os


def bar_plot(success_rate, Nifg, v, j):
    # 柱状图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.bar(Nifg, success_rate, width=5, align="edge")
    for x, y in zip(Nifg, success_rate):
        # ha: horizontal alignment（横向对齐方式）， va: vertical alignment（纵向对齐方式）
        # x+2.5, y+0.05 为标签的坐标
        plt.text(x + 2.5, y + 0.005, "%.2f" % y, ha="center", va="bottom")
    plt.title("Bar", fontsize=20)
    plt.xlabel("Nifg,v='%s' m" % v, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    # x轴刻度
    plt.xticks([index + 2.5 for index in Nifg], Nifg)
    plt.ylim(0, 1)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, "demo%s" % j))  # 分别命名图片


def bar_v(success_rate, v, j):
    # 柱状图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.bar(v, success_rate, width=0.005, align="edge")
    for x, y in zip(v, success_rate):
        # ha: horizontal alignment（横向对齐方式）， va: vertical alignment（纵向对齐方式）
        # x+2.5, y+0.05 为标签的坐标
        plt.text(x + 0.0025, y + 0.005, "%.2f" % y, ha="center", va="bottom")
    # plt.title("Bar", fontsize=20)
    plt.xlabel("v", fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    # x轴刻度
    plt.xticks([index + 0.0025 for index in v], v)
    plt.ylim(0, 1)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # x轴刻度保留两位小数
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, "demo%s" % j))  # 分别命名图片
