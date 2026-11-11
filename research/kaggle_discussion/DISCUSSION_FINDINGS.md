# Kaggle discussion findings: filament-segmentation-2026 (partial)

ESCALATE: the discussion forum itself could not be read. Kaggle pages are JS-rendered; WebFetch returns only the page title for the discussion index and for every notebook URL tried. The Kaggle CLI has no discussion command. Web search does not index the competition. What follows comes from local copies of public notebooks (research/kaggle_notebooks), which are participant writeups, not forum threads or host statements.

## Sources used
- research/kaggle_notebooks/please-fix-the-leaderboard/notebook_stripped.txt (artkomissar / "Data Shrushik", 2026-07-28)
- research/kaggle_notebooks/new-metric-solar-filament/notebook_stripped.txt (ektarr / Maxim, 2026-08-08)
- research/kaggle_notebooks/lb-0-70-topology-safe-solar-filaments-handoff/notebook_stripped.txt (phuongncn / Fususu, 2026-08-05)

## (1) Evaluator bug and metric change
- Participant claim (please-fix-the-leaderboard): "Kaggle returned Public Score 1.00" for a submission of 5 empty masks covering 5 of 180 test images. The author says 175 omitted images receive no penalty.
- Participant claim (new-metric-solar-filament): the competition evaluation page "changed the primary metric from Dice to Panoptic Quality (PQ) on 7 August 2026". This is quoted from the notebook. I did not verify it against the live evaluation page.

## (2) Is about 0.70 legitimate after the fix
- Participant claim (lb-0-70 handoff, Fususu): the exact 1,299-instance CSV "scored 0.70 twice". The notebook says the public LB "currently has a known empty/missing-prediction defect", so it is not treating 1.00 as valid.
- Timing caveat: this notebook is dated 2026-08-05, before the 2026-08-07 switch to PQ. Its 0.70 was likely scored under Dice, so it does not show a post-PQ 0.70.
- Methods named: Mask R-CNN + U-Net refiner (based on Nomannic19's notebook), refiner threshold 0.70, detector score 0.70, topology-safe 4-flip TTA, grouped folds by physical observation. The new-metric notebook uses PQ with IoU > 0.5 matching, 5-fold grouped CV, and medoid-annotator selection.

## (3) Host statements on test annotation
- None found. Nothing in the three notebooks is attributed to organizers. The "medoid annotator" and "multiple annotator records per observation" points are participant analysis of train annotations (new-metric notebook, lines 8-9 and 56-62). They say nothing about how the test set is annotated or which annotators are scored.

## (4) Post-processing tips reportedly raising the public score
- lb-0-70 handoff: threshold at 0.70 then merge positive overlaps, detector-seed split only when crowding gates allow it, and TTA accepted only if topology stays close to single-view. The author reports that many threshold, merge/split and consensus variants "tied 0.70" and that the next gain "probably needs better model evidence, not another broad post-processing sweep".
- No instances-per-image tip was found.

## Not done
- Forum thread titles, authors and dates; host announcements; post-PQ score reports. These need an authenticated browser session or a Kaggle endpoint I could not reach.
