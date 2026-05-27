================================================================================
RESEARCH DECISION: Should we continue pursuing 'StructMemory > TextBuffer'?
================================================================================

Decision rule:
  - If difference < 10pp and unstable: STOP pursuing structural difference
  - If difference >= 10pp and consistent across tasks: KEEP as secondary storyline

TextBuffer vs StructMemory differences (PO, retry=0):
  door-open-v3: TB=0.0%, SM=0.0%, diff=0.0pp
  pick-place-v3: TB=73.7%, SM=73.7%, diff=0.0pp
  push-v3: TB=66.7%, SM=64.7%, diff=2.0pp
  reach-v3: TB=69.7%, SM=69.7%, diff=0.0pp

TextBuffer vs StructMemory differences (Full Observable):
  pick-place-v3: TB=71.7%, SM=71.7%, diff=0.0pp
  push-v3: TB=73.9%, SM=73.9%, diff=0.0pp
  reach-v3: TB=80.0%, SM=80.0%, diff=0.0pp

Maximum difference across tasks: 2.0pp

DECISION: STOP: Difference < 10pp and/or unstable. Do NOT pursue 'StructMemory > TextBuffer' as main conclusion.

Recommended narrative shift:
  - Main conclusion A: Memory is necessary under PO
  - Main conclusion B: Naive retry mechanisms can be harmful
  - Main conclusion C: Recovery cost must be explicitly modeled