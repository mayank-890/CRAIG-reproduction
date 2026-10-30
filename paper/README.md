# Paper: CRAIG reproduction (IEEEtran conference)

Build (from this directory):

    python3 make_tables.py      # regenerates tables/*.tex from ../results/raw and ../results/processed
    ~/bin/tectonic paper.tex    # tectonic 0.15.0 (runs BibTeX automatically) -> paper.pdf

Figures in figures/ are copies of ../results/figures/*.png with the matplotlib titles cropped
(and two-panel plots split into single panels where noted); no data was re-plotted.
