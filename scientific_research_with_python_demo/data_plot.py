import matplotlib.pyplot as plt
import numpy as np
import os


def bar_plot(x, y, title, name, dx, x_name):
    # 柱状图
    plt.figure(figsize=(20, 8))
    # plt.rc("font", family="FangSong")
    plt.bar(x, y, width=0.5 * dx)
    for x0, y0 in zip(x, y):
        # ha: horizontal alignment（横向对齐方式）， va: vertical alignment（纵向对齐方式）
        # x+2.5, y+0.05 为标签的坐标
        if y0 != 0:
            plt.text(x0, y0 + 0.005, "%.2f" % y0, ha="center", va="bottom")
    plt.title(title, fontsize=16, y=1.05)
    # plt.xlabel("Nifg", fontsize=14)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    # x轴刻度
    plt.xticks(x)
    # plt.xticks(np.linspace(0.001, 0.2, 5) * 1000)
    plt.ylim(0, 1)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, name))  # 分别命名图片


def line_plot(x, y, title, name, x_name):
    # 折线图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.plot(x, y, linestyle="-")
    plt.title("Line", fontsize=16)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    plt.ylim(0, 1)
    # plt.xticks(x)
    # plt.xticks(np.linspace(0.001, 0.2, 5) * 1000)
    # plt.xticks([10, 30, 50, 70, 90, 110])
    plt.title(title, fontsize=20, y=1.05)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name))


def scatter_plot(est, desired, x_name, y_name, x_ax, tile, name):
    # 散点图
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.scatter(x_ax, est, marker=".", color="blue", linewidth=1, label="est")
    plt.title(tile, fontsize=20)
    plt.plot([0, np.max(x_ax)], [desired, desired], color="red", linestyle="--", linewidth=0.5, label="desired")
    plt.legend(loc="best")
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.ylim(-desired * 1.2, desired * 2)
    plt.margins(x=0)
    # plt.xticks(x_ax)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name))


def hist_plot(x, name, x_name, y_name, dx, title):
    # 绘制直方图
    plt.figure(figsize=(15, 8))
    # plt.rc("font", family="FangSong")
    nums, bins, patches = plt.hist(x, bins=dx, edgecolor="k", width=0.005)
    plt.xticks(bins, bins)
    for num, bin in zip(nums, bins):
        if num != 0:
            plt.annotate(num, xy=(bin, num), xytext=(bin, num + 0.5))
    plt.title(title, fontsize=20, y=1.05)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.xticks(np.linspace(-0.2, 0.2, 100))
    plt.xticks(rotation=50)
    plt.xlim(-0.2, 0.2)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name))


def mutil3_line(x, y, name, x_name, title):
    # 多条线并标签
    plt.figure()
    # plt.rc("font", family="FangSong")
    plt.plot(x, y[0], linestyle="-", label="SNR=10db")
    plt.plot(x, y[1], linestyle="-", label="SNR=30db")
    plt.plot(x, y[2], linestyle="-", label="SNR=70db")
    plt.title(title, fontsize=16)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(x)
    plt.legend(loc="best")
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name))


def mutil_line(x, y, name, x_name, title):
    # 多条线并标签
    plt.figure(figsize=(10, 8))
    # plt.rc("font", family="FangSong")
    plt.plot(x, y[0], linestyle="-", label="Nifg=10,dt=12*0.4d")
    plt.plot(x, y[1], linestyle="-", label="Nifg=20,dt=12*0.4d")
    plt.plot(x, y[2], linestyle="-", label="Nifg=30,dt=12*0.4d")
    plt.plot(x, y[3], linestyle="-", label="Nifg=10,dt=12*0.6d")
    plt.plot(x, y[4], linestyle="-", label="Nifg=20,dt=12*0.6d")
    plt.plot(x, y[5], linestyle="-", label="Nifg=30,dt=12*0.6d")
    plt.plot(x, y[6], linestyle="-", label="Nifg=10,dt=12*0.8d")
    plt.plot(x, y[7], linestyle="-", label="Nifg=20,dt=12*0.8d")
    plt.plot(x, y[8], linestyle="-", label="Nifg=30,dt=12*0.8d")
    plt.title(title, fontsize=16)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel("success_rate", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(x)
    # 图例显示到图外
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc=8, ncol=3)
    ax = plt.gca()
    # 坐标轴的边框（脊梁）去掉边框
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    figure_save_path = "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    plt.savefig(os.path.join(figure_save_path, name), bbox_inches="tight")
