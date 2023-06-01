import matplotlib.pyplot as plt
import numpy as np
import os


def bar_plot(x, y, name, dx, x_name):
    # 柱状图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.bar(x, y, width=0.5 * dx, align="edge")
    for x0, y0 in zip(x, y):
        # ha: horizontal alignment（横向对齐方式）， va: vertical alignment（纵向对齐方式）
        # x+2.5, y+0.05 为标签的坐标
        plt.text(x0 + 0.25 * dx, y0 + 0.005, "%.2f" % y0, ha="center", va="bottom")
    plt.title("Bar", fontsize=20)
    # plt.xlabel("Nifg", fontsize=14)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    # x轴刻度
    plt.xticks([index + 0.25 * dx for index in x], x)
    # plt.ylim(0, 1)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, name))  # 分别命名图片


def line_plot(x, y, name, x_name):
    # 折线图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.plot(x, y, linestyle="-.")
    plt.title("Line", fontsize=20)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    plt.ylim(0, 1)
    #
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name))
