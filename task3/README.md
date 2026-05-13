# Oceanus Folk — Rising Star Analysis Workbench

MC1 音乐知识图谱子集（Oceanus Folk）上的新星识别、多维刻画与交互可视化。离线管道产出 JSON，浏览器端用 D3.js 渲染。

## 仓库结构

| 路径 | 说明 |
|------|------|
| `MC1_graph.json` | 原始知识图谱数据 |
| `viz/preprocess_refactor.py` | 预处理脚本：图遍历、六维评分、导出 JSON |
| `viz/index_dense.html` | 主看板（`index.html` 会跳转至此） |
| `viz/data/` | 预处理生成的 JSON（已包含，可直接使用） |
| `viz/time_dim_charts.py` | 综合得分时间演化图生成脚本（依赖 `viz/data/artists.json`） |
| `viz/report_fig_t_trajectory.png` | 生成的论文图：综合得分随 career age 演化 |
| `viz/trajectory_tails.html` | 同一份数据的 D3.js 交互版 |
| `report/template.tex` | VGTC 期刊体例的论文源码 |

## 环境

- Python 3.x

## 快速开始

```bash
# 1. 重新生成数据（可选，viz/data/ 下已有现成数据可跳过）
cd viz
python preprocess_refactor.py

# 2. 启动本地服务
cd ..
python -m http.server 8080 --directory viz
# 浏览器访问 http://127.0.0.1:8080/index_dense.html
```

## 论文图：综合得分时间演化

`viz/time_dim_charts.py` 在 `viz/data/artists.json` 上复现项目原始的 9 个 rising-star 输入 + multiplicative composite 公式，并在每个 career age 上重新评估一次，得到综合得分随职业生涯演化的曲线。

```bash
python viz/time_dim_charts.py
# 输出：viz/report_fig_t_trajectory.png + viz/trajectory_tails.html
```

7 个被追踪的实体：Sailor Shift、Embers of Wrath、Orla Seabloom 三个 baseline；Ivy Echos 作为 Sailor Shift 前乐队的叙事背景；Copper Canyon Ghosts、Daniel O'Connell、Beatrice Albright 三个 predicted rising stars。Sailor Shift 终点 composite=0.4405 与 `viz/preprocess_refactor.py` 输出精确一致，可作为复现性的健全性检查。论文 LaTeX 源码在 `report/template.tex`，通过 `../viz/report_fig_t_trajectory.png` 引用上面这张图。

## 作者

维护者：[@14shi](https://github.com/14shi)
