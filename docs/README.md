# docs/ index — which document answers which question

| Question | Document |
|---|---|
| 为什么这个项目重要?整条故事链 + 全部实测证据(竞赛叙事) | `story_zh.md` |
| How is the model trained and tested, exactly (stages, losses, pixel roles, gates)? | `training_protocol.md` |
| How does cross-board transfer work; why context tokens; the board card? | `board_transfer_architecture.md` |
| What was measured, when, with what caveats (both sessions)? | `data_card.md` |
| Why is each design choice the way it is (dated log)? | `decisions.md` |
| What did each measurement exist for; what could be cut? | `measurement_targets.md` *(historical — fulfilled)* |
| The measurement/handover checklist we executed | `m3_data_requirements.md` *(historical — fulfilled)* |

Figures (`assets/`): `pipeline_overview.svg` (whole project, one glance) ·
`story_chain_zh.svg` (叙事链) · `operator_v25.svg` (architecture, exact shapes) ·
`training_scheme_v25.svg` (one training item). Result figures live in
`reports/qc/`.
