# Engine evaluation on SafeChat Lab collected data

Export: `pilot_bots_export.zip` · content model: distilbert_onnx

## Dataset

- **messages**: 1024
- **sessions**: 10
- **participants**: 17
- **humans**: 0
- **bots**: 18
- **pairs**: 63
- **positive pairs**: 4
- **hostile message rate**: 0.2559
- **target method**: {'broadcast': 580, 'reply': 257, 'mention': 187}
- **mean sessions per positive pair**: 9.25

## Pair-level detection

| System | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|---|---|
| stateless | 0.979 | 0.747 | 0.085 | 1.000 | 0.157 | 47 |
| frequency | 1.000 | 1.000 | 0.182 | 1.000 | 0.308 | 22 |
| engine | 1.000 | 1.000 | 0.571 | 1.000 | 0.727 | 7 |

## Streaming detection

| System | Harasser pairs detected | Median sessions to detect | False-alarm pairs |
|---|---|---|---|
| stateless | 100.0% | 1.0 | 43 |
| frequency | 100.0% | 1.5 | 18 |
| engine_medium_or_above | 100.0% | 3.0 | 8 |
| engine | 100.0% | 3.0 | 8 |

## Tier by role (pairs)

| Role | Low | Medium | High | Critical |
|---|---|---|---|---|
| banter | 2 | 0 | 0 | 2 |
| burst | 2 | 0 | 0 | 0 |
| escalator | 0 | 0 | 1 | 1 |
| harasser | 2 | 0 | 0 | 2 |
| normal | 33 | 0 | 0 | 0 |
| subtle_harasser | 1 | 0 | 0 | 1 |
| troll | 16 | 0 | 0 | 0 |

_Small samples: report counts and confidence intervals, and treat this as validation of the synthetic results rather than a replacement._
