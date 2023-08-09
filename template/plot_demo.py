import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import matplotlib.pyplot as plt
import csv


success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/H0.1success_SNR70nifg_20.csv", delimiter=",")
# success_rate = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_SNR50nifg_20.csv", delimiter=",")
Nifg_orig = np.arange(10, 101, 1)
# v_orig = np.arange(1, 201, 1) * 0.001
dt_orig = np.arange(1, 501, 1) * 0.01
# h_orig = np.arange(1, 151, 1)
h_orig = np.arange(1, 1501, 1) *0.1
# dp.line_plot(Nifg_orig, success_rate, "SNR=40,dt=12,v=0.005,h=30", "Nifg_SNR40_10_100_line", "Nifg")
# dp.line_plot(v_orig * 1000, success_rate, "SNR=50,dt=12,h=30,nifg=20", "V_50nifg_20", "v[mm/year]")
# dp.line_plot(dt_orig * 12, success_rate, "Nifg=20,SNR=70db,v=0.005,h=30", "dT_70nifg_20", "dt/day")
dp.line_plot(h_orig, success_rate, "SNR=70,dt=12,v=0.005,nifg=20", "H0x1_70nifg_20", "h/m")

# # 四条线绘制一张图
# y1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifg_success_SNR60_10_100.csv", delimiter=",")
# y2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifg_success_SNR50_10_100.csv", delimiter=",")
# y3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifg_success_SNR40_10_100.csv", delimiter=",")
# y4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifg_success_SNR30_10_100.csv", delimiter=",")
# # 将y1,y2,y3,y4以及对应的Nifg绘制在一张图上
# plt.figure()
# plt.plot(Nifg_orig, y1, label="SNR=60dB")
# plt.plot(Nifg_orig, y2, label="SNR=50dB")
# plt.plot(Nifg_orig, y3, label="SNR=40dB")
# plt.plot(Nifg_orig, y4, label="SNR=30dB")
# plt.xlabel("Nifg")
# plt.ylabel("success rate")
# plt.title("dT=12,v=0.005m,h=30m")
# # 根据SNR给四条线加标签
# plt.legend(loc="upper right")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/Nifg_SNR_10_100_line.png")

# V三条线绘制一张图
# v_orig = np.arange(1, 201, 1) * 0.001
# v1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_SNR70nifg_20.csv", delimiter=",")
# v2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_SNR60nifg_20.csv", delimiter=",")
# v3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_SNR50nifg_20.csv", delimiter=",")
# # 将y1,y2,y3以及对应的v绘制在一张图上
# plt.figure()
# plt.plot(v_orig * 1000, v1, label="SNR=70dB")
# plt.plot(v_orig * 1000, v2, label="SNR=60dB")
# plt.plot(v_orig * 1000, v3, label="SNR=50dB")
# plt.xlabel("v[mm/year]")
# plt.ylabel("success rate")
# plt.title("Nifg=20,dt=12,h=30m")
# # 根据SNR给三条线加标签
# plt.legend(loc="upper right")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/V_SNR_0_0.2_line.png")

# # h三条线绘制一张图
# h_orig = np.arange(1, 151, 1)
# h1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_SNR70nifg_20.csv", delimiter=",")
# h2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_SNR60nifg_20.csv", delimiter=",")
# h3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Hsuccess_SNR50nifg_20.csv", delimiter=",")
# # 将h1,h2,h3以及对应的h绘制在一张图上
# plt.figure()
# plt.plot(h_orig, h1, label="SNR=70dB")
# plt.plot(h_orig, h2, label="SNR=60dB")
# plt.plot(h_orig, h3, label="SNR=50dB")
# plt.xlabel("h[m]")
# plt.ylabel("success rate")
# plt.title("Nifg=20,dt=12,v=0.005m")
# # 根据SNR给三条线加标签
# plt.legend(loc="upper right")
# plt.savefig("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/plot/H_SNR_10_150_line.png")

