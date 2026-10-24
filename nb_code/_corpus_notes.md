# Corpus notes (generated for graphify semantic extraction)

## Tabular data (data/*.csv) — .csv is not parsed by graphify AST, schemas recorded here
- `data/chat_history.csv`: 14573 rows x 4 cols; columns: user_id, chat_id, timestamp, user_chat_text
  size: 2.1 MB
- `data/modules_catalog.csv`: 17 rows x 4 cols; columns: module_id, module_name, description_and_syllabus, prerequisite_level
  size: 0.0 MB
- `data/sample_submission.csv`: 1000 rows x 18 cols; columns: user_id, M_001, M_002, M_003, M_004, M_005, M_006, M_007, M_008, M_009, M_010, M_011, M_012, M_013, M_014, M_015, M_016, M_017
  size: 0.1 MB
- `data/test.csv`: 1000 rows x 1 cols; columns: user_id
  size: 0.0 MB
- `data/train_relevance.csv`: 4000 rows x 18 cols; columns: user_id, M_001, M_002, M_003, M_004, M_005, M_006, M_007, M_008, M_009, M_010, M_011, M_012, M_013, M_014, M_015, M_016, M_017
  size: 0.3 MB
- `data/user_assessments.csv`: 5000 rows x 2 cols; columns: user_id, assessment_result
  size: 5.4 MB

## Notebook code (notebooks/*.ipynb, 68 files, mirrored as nb_code/*.py for AST)
- 68 notebooks: v1, v10, v10_1, v11, v12, v12_1, v12_2, v13, v13_1, v14, v15, v16, ... (+56 more)
- Naming: v<N> = experiment version; suffix _1/_2 = variant; v42_1/v43_1 = numbered variants.

## Experiment results (results/v*)
- 66 version dirs: v1, v10, v10_1, v11, v12, v12_1, v12_2, v13, v13_1, v14 ... (+56 more)
- Each holds v*_summary.json (metrics/config of that experiment version); v49 adds reproducibility_config.json.

## Notes (sum/)
- sum/MineToday_V1-V46_EDA_Research_Gaps_and_V47_Architecture.md: architecture/research-gaps doc covering V1-V46 EDA and V47 design.
- sum/*.zip: chat handoff archives (not parsed).
