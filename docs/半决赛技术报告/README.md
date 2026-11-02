# 半决赛技术报告

`report.tex` 为源文件，`report.pdf` 为成稿。`figures` 保存流程图、架构图和模块图的 draw.io 源文件及 PDF。

使用 draw.io Desktop 修改图源并导出同名 PDF，导出选项为 `-x -f pdf --crop -b 10`。

在本目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
python finalize_pdf_links.py
python check_report_text.py
```

依赖为 TeX Live、Python、PyMuPDF 和 pypdf。链接处理脚本在每次重新编译后执行一次。

报告对应 Round327，实验依据见 `../../experiments/reports/round326_result.md`、`../../experiments/reports/round327_result.md`，训练实现见 `../../method/`。
