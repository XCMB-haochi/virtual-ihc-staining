# Report revision review

## Requirements checked

- The title block contains all three members: 团队成员、团队成员、团队成员. The date and standalone score banner are removed.
- Black headings, a single-column body, numbered equations, restrained figure colors and booktabs tables form the report layout.
- All 26 numbered equations are followed immediately by symbol explanations. The forward description covers input context, encoder, block operations, conditions, FiLM, decoder, output and crop.
- The parameter table reports disjoint module counts summing to 31,753,556; the existing strict-checkpoint audit and forward-shape record are retained.
- Internal round identifiers and version columns are absent from the reader-facing report.
- Thirteen references identify the architecture, normalization, initialization, loss, optimization and prediction-transfer methods.
- Performance analysis uses existing Liver results. Grouped teacher validation and full-data student platform results remain distinct.
- Both diagrams now use English labels and manuscript symbols. The overview traces ROI input, context extraction, encoder features e1--e4, conditioned bottleneck, decoder features d3--d1, output head, crop and CD45RO selection.
- The second figure separately expands the ConvNeXt V2 residual block, skip-feature concatenation and decoding, and the two-input FiLM computation.

## Original figures actually inspected

1. Cytoland, Extended Data Fig. 1, PDF page 17: dashed module boundaries, tensor dimensions, encoder/decoder separation and skip connections. Source: https://doi.org/10.1038/s42256-025-01046-2 ; publisher article PDF accessed through the German National Library copy https://d-nb.info/1374841188/34 .
2. ConvNeXt V2, Fig. 5, PDF page 6: vertical block operations and exterior identity path into the residual sum. Source: https://arxiv.org/pdf/2301.00808 .
3. U-Net, Fig. 1, PDF page 2: matched feature scales and lateral skip connections. Source: https://arxiv.org/pdf/1505.04597 .

The figures are native editable draw.io diagrams for this model, with its actual two-dimensional tensors and conditioning branches. They are exported by draw.io Desktop 31.7.0 as vector PDFs. Reference figures are used for visual organization, not inserted into the report. Edge labels are transparent separate text objects positioned beside the lines.

The overview uses dashed rounded module boundaries. In the ConvNeXt V2 panel, an explicit junction below the input formula connects both the convolution path and the residual path, independently of the formula label's bounding box. Citation frames are green, internal cross-reference frames red, and external URL frames cyan.

## Verification

The user lifted the earlier image-viewing restriction. The single-column report is inspected after rendering, including both draw.io figures, equation numbering, tables, title block and bibliography. The current link counts and resolved destinations are recorded in `report_link_audit.json`; external bibliography links remain usable. Thin visible frames mark link areas. Compilation diagnostics and PDF text/bounds checks supplement visual review. Generated compilation and preview files are cleaned after copying the final PDF to the delivery directory.

Draft-status wording, including the pending student-validation sentence, prospective experiment suggestions and internal delivery-verification prose, has been removed. Actual ROI variation, computational costs and evaluation scope remain factual parts of the report.

## Narrative and evidence review (2026-10-05)

The abstract and opening section now trace ROI grouping, domain pretraining, supervised fold teachers, held-out target generation, full-data student training, and fixed-model inference. Section introductions and architecture transitions distinguish the single forward pass from parameter learning and multi-view aggregation. Encoder depth-dependent learning rates are distinguished from the two supervised training phases. Section 7 separates method characteristics from limitations in connected prose.

Recipe and result statements were cross-checked against `tools/submission_round245/configs/recipe.json`, the packaged model/training source, and the Round217, Round239, Round243 and Round245 reports. Teacher OOF and full-data student platform results retain separate scopes. Largest fold/ROI decreases are described as changes, rather than mislabeled as the lowest-scoring fold/ROI. Rounded platform metrics are not used to attribute the small score difference.

Both existing draw.io diagrams were visually inspected and retained. The final ten-page PDF was rendered and inspected page by page for figure legibility, equation definitions, tables, section transitions and references. Text and link checks passed: 26 equations with definitions, 13 bibliography entries, and 58 links (45 internal, 13 external). No model, training recipe, prediction or submission archive was modified.

## Distillation and real-image figure completion (2026-10-05)

This revision supersedes the preceding ten-page/two-figure snapshot. The abstract, introduction and training section explicitly name knowledge distillation. The packaged `train_stage.py` confirms a 0.8 label-loss / 0.2 CD45RO teacher-loss mixture in student stage 2.

The four editable figures now cover the training/deployment overview, compact U-shaped architecture, internal blocks, and real held-out qualitative examples. Cytoland Figure 1 informs the separated pretraining, learning and image-example panels; references are not inserted into our figures. Sample identities, image hashes, teacher checkpoint hashes, fold exclusions, display window and crop positions are recorded in `figures/image_provenance.json`. Eleven source-image hashes and the fold/model metadata were rechecked. Teacher-cache outputs are identified as teacher predictions, separately from the final student's actual test output. No claim of independent student validation is made.

All twelve rendered pages were inspected for figure labels, arrows, images, captions, equations, tables, page breaks and bibliography. The qualitative caption was corrected to describe columns from left to right. Compilation/text checks cover 26 equations and 13 references; the link audit records 48 valid internal links and 13 external links. The delivery PDF is copied only after final verification. Training code, model parameters and submitted predictions are unchanged.

## Figure annotation refinement (2026-10-05)

Removed the overview's two prose notes and the architecture's repeated tensor-size note; the corresponding explanations remain in the text/captions. Added three thick dashed rounded panel boundaries to Figure 1, retaining module titles, input/output labels and the distillation-loss connection. Re-exported both changed figures from draw.io. The report remains twelve pages; pixel comparison against the previous delivery identifies only pages 2 and 3 as changed, and both were visually inspected after compilation. Text/reference and link checks passed.
