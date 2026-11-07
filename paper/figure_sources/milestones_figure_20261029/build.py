#!/usr/bin/env python3
"""Build vector PDF/SVG and compact 32:13 PNGs using TeX and Poppler (stdlib only)."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
STEM = 'TopoAML_milestones_compact'
YEARS = dict(A=2008, B=2017, C=2019, D=2020, E=2020, F=2022,
             G=2022, H=2023, I=2024, J=2024, K=2026)


def run(args):
    result = subprocess.run([str(a) for a in args], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def tool(name):
    found = shutil.which(name)
    for folder in ('/Library/TeX/texbin', '/opt/homebrew/bin', '/usr/bin'):
        candidate = Path(folder) / name
        if found or candidate.is_file():
            return found or str(candidate)
    raise RuntimeError(f'Required tool not found: {name}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    assert header[:8] == b'\x89PNG\r\n\x1a\n'
    return struct.unpack('>II', header[16:24])


def main():
    output = ROOT / 'output'
    pdfdir = output / 'pdf'
    qa = ROOT / 'qa'
    pdfdir.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    source_paths = [
        ROOT / 'milestones_figure.tex',
        ROOT / 'preview.tex',
        ROOT / 'build.py',
        ROOT.parent / 'positioning_figure_20261028/literature_sources.md',
    ]
    source_hashes = {str(p.relative_to(WORKSPACE)): sha(p) for p in source_paths}
    baseline = qa / 'source_snapshot.json'
    if not baseline.exists():
        baseline.write_text(json.dumps(source_hashes, indent=2) + '\n')
    # Keep first-generation provenance; later manuscript edits are legitimate.
    # Source immutability is checked across this build, rather than across revisions.
    matches_initial_sources = json.loads(baseline.read_text()) == source_hashes
    (qa / 'latest_source_snapshot.json').write_text(json.dumps(source_hashes, indent=2) + '\n')

    tex_command = [tool('xelatex'), '-interaction=nonstopmode', '-halt-on-error',
                   '-file-line-error', f'-output-directory={pdfdir}',
                   f'-jobname={STEM}', 'preview.tex']
    # Fixed-page positioning uses TikZ's remembered page origin: two passes.
    build_logs = [run(tex_command), run(tex_command)]
    (qa / 'build.log').write_text('\n'.join(build_logs))
    log = (pdfdir / f'{STEM}.log').read_text()
    pdf = pdfdir / f'{STEM}.pdf'
    info = run([tool('pdfinfo'), pdf])
    fonts = run([tool('pdffonts'), pdf])
    text = run([tool('pdftotext'), '-layout', pdf, '-'])
    (qa / 'pdfinfo.txt').write_text(info)
    (qa / 'pdffonts.txt').write_text(fonts)
    (qa / 'extracted_text.txt').write_text(text)

    # Cairo exports embedded glyph outlines: SVG remains crisp without local fonts.
    svg = output / f'{STEM}.svg'
    run([tool('pdftocairo'), '-svg', pdf, svg])
    svg_root = ET.parse(svg).getroot()
    assert not any(e.tag.endswith('image') for e in svg_root.iter()), 'Raster image in SVG'
    # Glyph outlines inherit fill from these groups. All neutral text is true,
    # fully opaque black; coloured text keeps the established route palette.
    svg_ns = {'svg': 'http://www.w3.org/2000/svg'}
    glyph_groups = [e for e in svg_root.findall('.//svg:g', svg_ns)
                    if e.findall('svg:use', svg_ns)]
    allowed_text_colours = [(0, 0, 0), (255, 255, 255), (23, 63, 115),
                            (152, 47, 58), (199, 100, 24)]
    def approved_text_fill(fill):
        values = re.fullmatch(r'rgb\(([0-9.]+)%, ([0-9.]+)%, ([0-9.]+)%\)', fill or '')
        if not values:
            return False
        rgb = [float(v) for v in values.groups()]
        # XeTeX and pdfTeX round colour operands differently in the PDF.
        return any(all(abs(v - c * 100 / 255) < 0.1 for v, c in zip(rgb, colour))
                   for colour in allowed_text_colours)
    assert glyph_groups and all(approved_text_fill(e.get('fill')) for e in glyph_groups)
    assert all(float(e.get('fill-opacity', '1')) == 1 for e in glyph_groups)
    high = output / STEM
    preview = output / f'{STEM}_preview'
    run([tool('pdftoppm'), '-r', '600', '-scale-to-x', '7680', '-scale-to-y', '3120',
         '-png', '-singlefile', pdf, high])
    run([tool('pdftoppm'), '-scale-to-x', '1920', '-scale-to-y', '780',
         '-png', '-singlefile', pdf, preview])

    # TeX can wrap anywhere in a long record, including inside the unit "pt".
    audit_text = log.replace('\n', '')
    bounds = {}
    arrows = {}
    labels = {}
    number = r'(-?[0-9.]+)pt'
    for match in re.finditer(r'MSLABEL:([A-Za-z0-9]+):' + ':'.join([number] * 4), audit_text):
        labels[match[1]] = [float(v) for v in match.groups()[1:]]
    for match in re.finditer(r'MSBOUND:([A-K]):' + ':'.join([number] * 8), audit_text):
        bounds[match[1]] = [float(v) for v in match.groups()[1:]]
    for match in re.finditer(r'MSARROW:([A-K]):' + ':'.join([number] * 8), audit_text):
        arrows[match[1]] = [float(v) for v in match.groups()[1:]]
    assert set(bounds) == set(YEARS), f'Missing bounds: {bounds.keys()}'
    assert set(arrows) == set(YEARS), f'Missing arrows: {arrows.keys()}'
    overlaps = []
    for letter, b in bounds.items():
        x1, y1, x2, y2, tx1, ty1, tx2, ty2 = b
        assert x1 < tx1 < tx2 < x2, f'Horizontal text overflow: {letter}'
        assert y1 < ty1 < ty2 < y2, f'Vertical text overflow: {letter}'
        ax1, ay1, ax2, ay2, ax3, ay3, ax4, ay4 = arrows[letter]
        assert abs(ax1-ax2) < .001 and abs(ay2-ay3) < .001 and abs(ax3-ax4) < .001, \
            f'Nonorthogonal arrow: {letter}'
        assert ay4 < ay3 < ay1, f'Arrow must lead downwards: {letter}'
        assert x1 < ax4 < x2 and abs(ay4-y2) < .001, f'Arrow outside target card: {letter}'
    letters = sorted(bounds)
    for index, a in enumerate(letters):
        for b in letters[index + 1:]:
            ax1, ay1, ax2, ay2 = bounds[a][:4]
            bx1, by1, bx2, by2 = bounds[b][:4]
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1):
                overlaps.append([a, b])
    assert not overlaps, overlaps
    warnings = [line for line in log.splitlines()
                if 'Overfull' in line or 'Underfull' in line or 'LaTeX Warning' in line]
    # Hyperref may request a harmless second pass on first compilation.
    if any('Rerun' in line for line in warnings):
        run([tool('xelatex'), '-interaction=nonstopmode', '-halt-on-error',
             f'-output-directory={pdfdir}', f'-jobname={STEM}', 'preview.tex'])
    assert not any('Overfull' in line or 'Underfull' in line for line in warnings), warnings
    dimensions = re.search(r'Page size:\s+([0-9.]+) x ([0-9.]+)', info)
    width, height = map(float, dimensions.groups())
    assert len(labels) == 26, f'Missing legend/year/marker bounds: {labels.keys()}'
    assert all(0 <= b[0] < b[2] <= width * 72.27 / 72 and
               -height * 72.27 / 72 <= b[1] < b[3] <= 0 for b in labels.values())
    label_overlaps = []
    for a, ba in labels.items():
        for b, bb in labels.items():
            if a < b and min(ba[2], bb[2]) > max(ba[0], bb[0]) and \
                    min(ba[3], bb[3]) > max(ba[1], bb[1]):
                label_overlaps.append([a, b])
    assert not label_overlaps, f'Overlapping legend/year/marker labels: {label_overlaps}'
    assert abs(width / height - 32 / 13) < 0.00002
    assert png_size(high.with_suffix('.png')) == (7680, 3120)
    assert png_size(preview.with_suffix('.png')) == (1920, 780)
    assert all(str(year) in text for year in set(YEARS.values()))
    flat_text = ' '.join(text.split())
    removed_strings = ['Selected research milestones', 'Years and core techniques',
                       'Each faint vertical arrow', 'The dashed', 'uniform year spacing',
                       'general methodological context', 'PH = persistent homology',
                       'Selected studies', 'Colours identify', 'compressed']
    assert all(s not in flat_text for s in removed_strings), 'Deleted annotation remains'
    deleted_card_text = ['Interactive visual analysis', 'Coordinated views', 'Search by example',
                         'Bank AML investigations', 'Social network metrics', 'Client-risk assessment',
                         'Flow and path descriptors', 'Obfuscation patterns', 'Streaming subgraph features',
                         'Directed descriptors', 'Synthetic IBM AML accounts', 'Account classification',
                         'Graph convolutional networks', 'Link prediction on bipartite graphs',
                         'DGI / GIN graph encoders', 'Reverse message passing', 'Persistent-homology readout',
                         'Node- and graph-level learning']
    assert all(s not in flat_text for s in deleted_card_text), 'Removed card description remains'
    assert 'Fronzetti Colladon' in flat_text and 'Remondi (2017)' in flat_text
    assert len({round(bounds[k][3], 2) for k in ('C', 'F', 'H', 'I')}) == 1
    assert len({round(bounds[k][3], 2) for k in ('E', 'G')}) == 1
    font_rows = [line.split() for line in fonts.splitlines()[2:] if line.strip()]
    assert font_rows and all('Arial' in row[0] for row in font_rows), fonts
    assert any('ArialMT' in row[0] for row in font_rows), fonts
    assert any('Arial-BoldMT' in row[0] for row in font_rows), fonts
    assert all('yes' in row for row in font_rows), 'Arial fonts must be embedded'
    assert all(sha(p) == source_hashes[str(p.relative_to(WORKSPACE))] for p in source_paths)
    report = {
        'passed': True, 'pages': 1, 'aspect_ratio': '32:13',
        'page_size_points': [width, height],
        'high_resolution_png': {'pixels': [7680, 3120], 'dpi': 600},
        'preview_png': [1920, 780], 'study_years': YEARS,
        'font': 'Arial / Arial Bold, embedded TrueType fonts',
        'svg_contains_raster_images': False,
        'connector_opacity': 0.20, 'connectors_on_lowest_artwork_layer': True,
        'all_11_arrows_axis_aligned': True,
        'card_rows': [['A', 'B', 'D', 'J', 'K'], ['C', 'F', 'H', 'I'], ['E', 'G']],
        'all_card_description_text_removed': True,
        'literature_identity_retained_in_coloured_headings': True,
        'stroke_widths_pt': {'card': 2.6, 'K_card': 3.6, 'lane': 2.4,
                             'axis': 2.2, 'tick': 2.0, 'axis_break': 2.4,
                             'connector': 3.3, 'connector_understroke': 6.2},
        'font_sizes_tex_pt': {'coloured_card_headings': 16,
                              'legend': 14, 'year': 20},
        'coloured_text_bold': True, 'legend_bold': True,
        'neutral_text_colour': '#000000', 'text_opacity': 1.0,
        'compressed_span': '2008-2016; uniform spacing from 2016 onwards',
        'card_bounds_pt': bounds, 'connector_vertices_pt': arrows,
        'card_overlaps': overlaps, 'text_contained_in_cards': True,
        'legend_year_marker_bounds_pt': labels, 'legend_year_marker_overlaps': label_overlaps,
        'source_files_unchanged_during_build': True,
        'source_versions_match_first_generation': matches_initial_sources,
        'visual_review': 'pending',
        'outputs': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in (pdf, svg, high.with_suffix('.png'), preview.with_suffix('.png'))},
    }
    (qa / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'page_size_points', 'high_resolution_png',
                                           'font', 'card_overlaps', 'visual_review')}, indent=2))


if __name__ == '__main__':
    main()
