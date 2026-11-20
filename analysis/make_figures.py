"""Rebuild the result figures of the paper (Figures 6-17: ROC curves, Critical Difference diagrams and accuracy/AUC
scatter plots) as TikZ/pgfplots code, in the black-and-white style of the manuscript.

Usage (from the repository root, after running the experiments):
    python analysis/make_figures.py          # writes results/figures/*.tex (+ figures.pdf if pdflatex is installed)

Each figure is written as a TikZ fragment that can be \\input in a LaTeX document whose preamble contains the pgfplots
styles of results/figures/styles.tex. results/figures/figures.tex collects all the fragments in one standalone document.
In the manuscript, the labels of the scatter plots were placed manually to avoid overlaps; here they are placed
automatically, so they may overlap.
"""
import sys, pathlib, shutil, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.config import RESULTS
from src.results import load_all, friedman_nemenyi, ranking_candidates, CLASSICAL, STATIC

OUT = RESULTS / 'figures'
GRID = np.linspace(0, 1, 21)
STYLES = (r'\pgfplotsset{every axis/.append style={legend cell align=left}, '
          + ', '.join(f'sty{i + 1}/.style={{{c}, {ls}, mark={mk}, mark options={{solid}}}}' for i, (c, ls, mk) in enumerate([
              ('black', 'solid', 'o'), ('black', 'dashed', 'square'), ('black', 'dotted', 'triangle'), ('black', 'dashdotted', 'diamond'),
              ('gray!70!black', 'solid', 'x'), ('gray!70!black', 'dashed', '+'), ('gray!70!black', 'dotted', 'star'),
              ('gray!70!black', 'dashdotted', 'pentagon'), ('black', 'densely dashed', '*'), ('black', 'loosely dotted', 'square*'),
              ('gray!70!black', 'densely dashed', 'triangle*'), ('gray!70!black', 'loosely dashed', 'diamond*'),
              ('black', 'densely dotted', 'otimes')])) + '}\n')


def roc_tex(fam_results, names):
    """Mean ROC curve of each model over the splits (TPR interpolated on a common FPR grid).
    Models with identical curves (e.g. a DES method and its FIRE variant) are drawn once with a merged label."""
    curves = {}
    for n in names:
        m = np.mean([np.interp(GRID, f, t) for f, t in fam_results[n]['roc']], 0)
        m[0], m[-1] = 0, 1
        key = (tuple(np.round(m, 6)), round(float(np.mean(fam_results[n]['auc'])), 4))
        curves.setdefault(key, []).append(n)
    lines = []
    for i, ((m, auc), ns) in enumerate(curves.items()):
        coords = ' '.join(f'({x:.2f},{y:.3f})' for x, y in zip(GRID, m))
        lines.append(f'\\addplot[sty{i % 13 + 1}, thick, mark size=1.3pt, mark repeat=2] coordinates {{{coords}}};\n'
                     f'\\addlegendentry{{{" / ".join(ns)} (AUC = {auc:.3f})}}')
    return ('\\begin{tikzpicture}\n\\begin{axis}[width=12cm, height=8cm, xmin=0, xmax=1, ymin=0, ymax=1.02, grid=major, '
            'xlabel={False Positive Rate}, ylabel={True Positive Rate}, legend pos=south east, '
            'legend style={font=\\small, fill opacity=0.9}, label style={font=\\small}, tick label style={font=\\small}]\n'
            + '\n'.join(lines) + '\n\\addplot[gray, dashed, thick, forget plot] coordinates {(0,0) (1,1)};\n\\end{axis}\n\\end{tikzpicture}\n')


def cd_tex(scores, metric):
    """Demsar-style Critical Difference diagram (Friedman test + Nemenyi post-hoc test, alpha = 0.05)."""
    t_ = friedman_nemenyi(scores)
    names = list(scores); ranks = np.array([t_['ranks'][n] for n in names]); cd = t_['cd']
    k = len(names); order = np.argsort(ranks)
    W = 12.0; x = lambda r: 1.0 + (W - 2.0) * (r - 1) / (k - 1)
    t = ['\\begin{tikzpicture}[font=\\normalsize]', f'\\draw[thick] ({x(1):.3f},0) -- ({x(k):.3f},0);']
    for i in range(1, k + 1):
        t.append(f'\\draw ({x(i):.3f},0) -- ({x(i):.3f},0.15) node[above] {{{i}}};')
    t.append(f'\\draw[very thick] ({x(1):.3f},0.9) -- ({min(x(1 + cd), x(k)):.3f},0.9);')
    t.append(f'\\node[above] at ({(x(1) + min(x(1 + cd), x(k))) / 2:.3f},0.9) {{CD = {cd:.3f}}};')
    half = (k + 1) // 2
    for j, idx in enumerate(order):
        r = ranks[idx]; y = -0.45 - 0.42 * (j if j < half else (k - 1 - j))
        side = (0.6, 'left') if j < half else (W - 0.6, 'right')
        t.append(f'\\draw ({x(r):.3f},0) |- ({side[0]},{y:.3f}) node[{side[1]}] {{{names[idx]} ({r:.2f})}};')
    sr = ranks[order]; cliques = []
    for i in range(k):  # groups of models whose ranks differ by less than the CD
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        t.append(f'\\draw[line width=2.5pt] ({x(sr[i]) - 0.08:.3f},{-0.18 - 0.12 * c:.3f}) -- ({x(sr[j]) + 0.08:.3f},{-0.18 - 0.12 * c:.3f});')
    p = '$p < 0.001$' if t_['p'] < 0.001 else f"$p = {t_['p']:.3f}$"
    t.append(f'\\node[above] at ({W / 2:.3f},1.35) {{Friedman ({metric}): $\\chi^2$ = {t_["stat"]:.3f}, {p}}};')
    return '\n'.join(t) + '\n\\end{tikzpicture}\n'


def scatter_tex(F):
    groups = [('DES (Pool 2)', 'mark=*, mark size=2.5pt', 'pool2'), ('Static ensembles', 'mark=square, mark size=2.5pt', 'static'),
              ('Classical', 'mark=triangle, mark size=2.8pt', 'classical')]
    t = ['\\begin{tikzpicture}\n\\begin{axis}[width=12cm, height=8cm, grid=major, xlabel={Accuracy}, ylabel={AUC-ROC}, '
         'legend pos=north west, legend style={font=\\small}, nodes near coords, point meta=explicit symbolic, '
         'every node near coord/.append style={font=\\scriptsize, anchor=west}]']
    for gname, mk, fam in groups:
        merged = {}
        for n, r in F[fam].items():
            merged.setdefault((round(float(np.mean(r['acc'])), 4), round(float(np.mean(r['auc'])), 3)), []).append(n)
        coords = ' '.join(f'({a:.4f},{u:.4f}) [{"/".join(ns)}]' for (a, u), ns in merged.items())
        t.append(f'\\addplot[only marks, {mk}, black] coordinates {{{coords}}};\n\\addlegendentry{{{gname}}}')
    return '\n'.join(t) + '\n\\end{axis}\n\\end{tikzpicture}\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'styles.tex').write_text(STYLES)
    A = load_all()
    files = []
    for fusion, first in [('early', 6), ('late', 12)]:
        F = A[fusion]
        for num, fam in enumerate(['classical', 'static', 'pool1', 'pool2', 'pool3'], first):
            names = CLASSICAL if fam == 'classical' else STATIC if fam == 'static' else list(F[fam])
            for sub, tex in [('a', roc_tex(F[fam], names)), ('b', cd_tex({n: F[fam][n]['acc'] for n in names}, 'accuracy'))]:
                name = f'fig{num:02d}{sub}_{fusion}_{fam}_{"roc" if sub == "a" else "cd"}'
                (OUT / f'{name}.tex').write_text(tex); files.append(name)
        num = first + 5
        top5 = ranking_candidates(F)[:5]
        for sub, kind, tex in [('a', 'cd_top5', cd_tex({n: r['auc'] for n, r in top5}, 'AUC-ROC')), ('b', 'scatter', scatter_tex(F))]:
            name = f'fig{num:02d}{sub}_{fusion}_{kind}'
            (OUT / f'{name}.tex').write_text(tex); files.append(name)
    doc = ('\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[margin=1.5cm]{geometry}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n'
           '\\input{styles.tex}\n\\begin{document}\n'
           + '\n'.join(f'\\section*{{{f.replace("_", " ")}}}\n\\resizebox{{\\linewidth}}{{!}}{{\\input{{{f}.tex}}}}\n\\clearpage' for f in files)
           + '\n\\end{document}\n')
    (OUT / 'figures.tex').write_text(doc)
    print(f'{len(files)} TikZ figures written to {OUT}')
    if shutil.which('pdflatex'):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'figures.tex'], cwd=OUT, capture_output=True, text=True)
        print('figures.pdf compiled' if r.returncode == 0 else 'pdflatex failed, see results/figures/figures.log')
        for ext in ['aux', 'log' if r.returncode == 0 else 'none']:
            (OUT / f'figures.{ext}').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
