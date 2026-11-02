# 复赛技术报告

`report.pdf` 为复赛方案报告，`report.tex` 为源文件。`figures/` 保存四张图的 draw.io 源文件与 PDF：训练流程、模型架构、基本模块及预测示例。

报告记录复赛完整方案及 79.1052 的平台成绩，内容包括上下文输入、空间条件、知识蒸馏和几何 TTA。对应实验见 `../../experiments/reports/`，复赛提交代码见 `../../experiments/tools/submission_round245/`。

在本目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
python finalize_pdf_links.py
python check_report_text.py
```

依赖 TeX Live、Python、PyMuPDF 和 pypdf。图源由 draw.io Desktop 编辑并导出，选项为 `-x -f pdf --crop -b 10`。链接处理脚本在每次重新编译后执行一次。
