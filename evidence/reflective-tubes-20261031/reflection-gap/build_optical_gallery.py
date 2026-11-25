#!/usr/bin/env python3
"""Render recorded optical decisions over their exact native photos. No acquisition."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CAPTURES = HERE.parent / 'photos-at-scouts'
spec = importlib.util.spec_from_file_location('validated_recaptures', CAPTURES / 'build_gallery.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
ORDER = (15, 6, 53, 8, 47)
NOTES = {
    15: 'Operator reports a reflection was selected. Which coexisting ridge is physical remains unconfirmed; a narrower optical trace is not proof of the physical laser path.',
    6: 'Operator identifies the apparent dip as a reflection and reports a large mating gap. A rejected image feature does not identify the physical material lips.',
    53: 'Operator identifies the apparent dip as a reflection and reports a large mating gap. Any gap center remains inferred unless both physical lips are independently resolved.',
    8: 'First positive-tilt failure. Compare whole-frame competing returns with the narrow search ROI; neither establishes a measured seam.',
    47: 'Operator identifies this laser return as valid but says the laser plane misses the joint. A valid single surface is insufficient for a two-lip gap or seam.',
}
TITLES = {15: 'Opposite scout · −5° / −3 mm', 6: 'Center scout · 0° / 0 mm',
          53: 'Negative edge scout · −10° / −9 mm', 8: 'Positive scout · +5° / +3 mm',
          47: 'Positive edge scout · +10° / +9 mm'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indices(values, n, label):
    require(isinstance(values, list) and all(type(i) is int and 0 <= i < n for i in values),
            f'{label}: invalid candidate-table index')
    require(len(set(values)) == len(values), f'{label}: duplicate candidate index')
    return set(values)


def validate_decision(item, snapshot, png_sha):
    """All decision indices belong to this decision's own native candidate table."""
    metadata = snapshot['metadata']
    require(item['png_sha256'] == png_sha, 'Decision PNG SHA256 mismatch')
    for key in ('calibration_id', 'optical_intrinsics_id'):
        require(item[key] == metadata[key], f'Decision {key} mismatch')
    cloud = item['cloud']
    for key in helpers.IDENTITY + ('optical_intrinsics_id', 'read_started', 'controls'):
        require(cloud.get(key) == metadata.get(key), f'Decision source identity differs: {key}')
    f = cloud['reflection']
    require(f.get('schema') == 'weldfinder.native_reflection.v1', 'Unknown optical decision schema')
    require(f.get('physical_primary_proven') is False, 'Gallery requires explicitly unproven physical primary')
    for key in helpers.IDENTITY + ('optical_intrinsics_id', 'read_started', 'controls'):
        require(f['identity'].get(key) == metadata.get(key), f'Candidate-table identity differs: {key}')
    table, size = f['candidate_table'], snapshot['image']['image_size']
    n = table['count']
    require(type(n) is int and n >= 0 and len(table['uv']) == n, 'Malformed candidate table count')
    require(all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float))
                and math.isfinite(v) for v in p) and 0 <= p[0] < size[0]
                and 0 <= p[1] < size[1] for p in table['uv']), 'Invalid native candidate UV')
    selected = indices(f['selected_candidate_indices'], n, 'Selected')
    unresolved = indices(f['unresolved_candidate_indices'], n, 'Unresolved')
    reject = set()
    reasons = {}
    for name, values in f['rejected_reason_indices'].items():
        reason_set = indices(values, n, name)
        reject |= reason_set
        reasons[name] = len(reason_set)
    require(not selected & (unresolved | reject), 'Selected candidate also rejected or unresolved')
    # Some reason sets intentionally overlap unresolved (e.g. ambiguous_competing_return).
    # Missing classifications after detector overflow are displayed as ambiguous, never accepted.
    unclassified = set(range(n)) - selected - unresolved - reject
    amber = unresolved | unclassified
    gray = reject - amber
    require(cloud['count'] == len(selected), 'Selected cloud count differs from decision indices')
    actual_selected_uv = sorted(tuple(p) for p in cloud['uv'])
    require(actual_selected_uv == sorted(tuple(table['uv'][i]) for i in selected),
            'Selected cloud UV differs from its candidate-table selections')
    groups = {'selected': selected, 'ambiguous': amber, 'rejected': gray}
    return {
        'status': f['status'], 'summary': f['summary'], 'processing_ms': item.get('processing_ms'),
        'physical_primary_proven': False, 'count': n,
        # Quantization is display-only; native source arrays and hashes remain unchanged.
        'uv': [[round(float(u), 2), round(float(v), 2)] for u, v in table['uv']],
        **{name: sorted(ids) for name, ids in groups.items()},
        'unclassified_displayed_ambiguous': len(unclassified), 'rejected_reason_counts': reasons,
    }


def gap_for(data, index, identity):
    """Optional evidence is shown verbatim, without inventing a spatial marker."""
    if data is None:
        return None
    rows = data if isinstance(data, list) else data.get('observations', data.get('results', []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    matches = [r for r in rows if r.get('observation') in (index, f'obs-{index:04d}')]
    if not matches:
        return None
    for row in matches:
        require(row.get('png_sha256', row.get('native_png_sha256')) == identity,
                'Gap inference lacks matching native photo hash')
    for row in matches:
        require(row.get('found') is False and row.get('optical_support', {}).get('qualified') is False,
                'This comparison requires rejected coupled seam decisions, not new measured seams')
        fitup = row.get('fitup_gap', {})
        require(fitup.get('counts_as_measured_seam') is False
                and fitup.get('counts_as_observed_angular_coverage') is False,
                'Gap diagnostics must explicitly reject measured-seam and angular-coverage claims')
    return matches


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reflection decisions on the same laser photos</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101722;color:#edf4fc;font:15px/1.45 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:22px}h1{font-size:29px;line-height:1.15}h2{margin:0;font-size:20px}p{color:#c1cfdf}.notice{background:#28303b;border-left:4px solid #ffc25b;padding:12px 16px}.controls{position:sticky;top:0;z-index:4;background:#111b27ef;padding:12px;display:flex;gap:14px;flex-wrap:wrap;border:1px solid #3b4d61}button,select{font:inherit;background:#243b51;color:white;border:1px solid #61819f;border-radius:5px;padding:6px 10px}button{cursor:pointer}.card{margin:22px 0;border:1px solid #40516a;border-radius:8px;overflow:hidden;background:#182431}.heading,.detail{padding:14px 16px}.small{font-size:12px;color:#aac0d5}.views{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#40516a}.view{min-width:0;background:#111b26}.caption{padding:8px 12px;color:#d5e2ef}svg.photo{width:100%;display:block;background:#03070a;touch-action:none;cursor:grab}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.green{color:#51ff81}.cyan{color:#37daff}.gray{color:#a8b2c0}.amber{color:#ffb73d}details{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;color:#b8cce0}code{overflow-wrap:anywhere}body.raw .marks{display:none}label{white-space:nowrap}table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}td,th{text-align:left;padding:8px;border-bottom:1px solid #40516a}.tablewrap{overflow:auto}.counts{font-size:13px;margin:6px 0}strong{color:#f5f8fc}.gap{padding:10px;border:1px solid #796435;background:#29291f}@media(max-width:650px){main{padding:12px 8px}.views{grid-template-columns:1fr}.controls{gap:8px}.heading{padding:12px}h1{font-size:25px}}</style>
<main><h1>Reflection handling: before and after, on the same photograph</h1><p>The left view shows the earlier accepted laser pixels. The right view shows the conservative optical decision for that <strong>exact same October 31 exposure</strong>.</p>
<div class="notice"><strong>Optical selection does not prove a physical primary laser return.</strong> Operator interpretations are labeled below. A missing interval can be glare, reflection, occlusion, poor signal or a physical gap. No inferred midpoint is counted as a measured weld seam. These new pose recaptures are not the original October 30 exposures or new angular-coverage measurements.<p id="stage"></p></div>
<div class="controls"><label>Scope <select id="scope"><option value="saved_roi">Saved search ROI</option><option value="full_frame">Full native image</option></select></label><button id="zoom">Show full photograph</button><button id="raw">Hide dots</button><label>Display gamma <input id="gamma" type="range" min=".2" max="1" step=".05" value=".5"><output id="gv">0.50</output></label></div>
<p class="legend"><span class="green">● Before: accepted returns</span><span class="cyan">● After: optically selected, physical identity unproven</span><span class="gray">● Rejected candidates</span><span class="amber">● Ambiguous / unclassified candidates</span></p><p class="small">Dashed rectangle: recorded search ROI. Both views share one source image. Display gamma 0.5 is applied before JPEG compression; native detection pixels are unchanged. Default view focuses on the ROI; drag to pan and use the mouse wheel or pinch to zoom. Native UV positions are not smoothed or extended.</p>
<svg width="0" height="0"><defs><filter id="gamma-filter" color-interpolation-filters="sRGB"><feComponentTransfer><feFuncR type="gamma" exponent="1"/><feFuncG type="gamma" exponent="1"/><feFuncB type="gamma" exponent="1"/></feComponentTransfer></filter></defs></svg>
<div class="notice"><strong>Current result: 0 qualified seams; no fitted tube model is accepted.</strong><div class="tablewrap" id="summary-table"></div></div><div id="cards">Loading verified evidence…</div><p class="small">Build checks: original PNG bytes and decoded pixel hashes, live optical calibration, exact decision/photo acquisition identity, candidate-table indices, and recapture measured endpoint proof. Reduced JPEGs are display copies. Original native evidence remains unchanged.</p></main>
<script id="evidence" type="application/octet-stream">PAYLOAD</script><script>
(async()=>{const bytes=Uint8Array.from(atob(document.querySelector('#evidence').textContent),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));const D=JSON.parse(await new Response(stream).text());const NS='http://www.w3.org/2000/svg',E=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let scope='saved_roi',focus=true;const live=[];const node=(name,attrs={})=>{const n=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;};
document.querySelector('#stage').textContent=D.prototype?'PROTOTYPE comparison — detector decisions are still under review.':'Offline candidate review — selection is an optical hypothesis, not physical seam qualification.';document.querySelector('#cards').textContent='';
const rows=D.cards.map(d=>{const a=d.gap?.[0];return `<tr><td>${String(d.index).padStart(4,'0')}</td><td>${E(d.decisions.saved_roi.status)}</td><td>${E(d.decisions.full_frame.status)}</td><td>${E(a?.status??'not evaluated')}</td><td>${E(a?.fitup_gap?.status??'not qualified')}</td><td>${E(a?.quality?.reason??'No coupled geometry evidence supplied')}</td></tr>`;}).join('');document.querySelector('#summary-table').innerHTML=`<table><thead><tr><th>Obs</th><th>ROI optics</th><th>Full-frame optics</th><th>Seam</th><th>Gap</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table>`;
function marks(d,after){const group=node('g',{class:'marks'}),decision=d.decisions[scope];const layers=after?[['rejected','#9ca9bc',.65],['ambiguous','#ffb73d',1.35],['selected','#37daff',1.9]]:[['before','#51ff81',1.9]];for(const[k,color,r]of layers){const uv=k==='before'?(scope==='saved_roi'?d.roi_points:d.context_points)||[]:decision[k].map(i=>decision.uv[i]);const g=node('g',{fill:color,opacity:k==='rejected'?.65:1});for(const[u,v]of uv)g.append(node('circle',{cx:u,cy:v,r}));group.append(g);}group.append(node('rect',{x:d.roi[0]*d.width,y:d.roi[1]*d.height,width:(d.roi[2]-d.roi[0])*d.width,height:(d.roi[3]-d.roi[1])*d.height,fill:'none',stroke:'#ffc25b','stroke-width':1.5,'stroke-dasharray':'7 5'}));return group;}
function photo(d,after){const s=node('svg',{class:'photo',role:'img','aria-label':`${after?'Conservative decisions':'Original accepted returns'}, observation ${d.index}`});s.append(node('image',{href:d.jpeg,width:d.width,height:d.height,filter:'url(#gamma-filter)'}));const state={s,d,after,box:d.crop.slice(),drag:null,touches:new Map()};live.push(state);function box(){s.setAttribute('viewBox',`${state.box[0]} ${state.box[1]} ${state.box[2]} ${state.box[3]}`);}state.reset=()=>{const b=focus?d.crop:[0,0,d.width,d.height];state.box=[b[0],b[1],b[2]-b[0],b[3]-b[1]];box();};state.paint=()=>{s.querySelector('.marks')?.remove();s.append(marks(d,after));};state.reset();state.paint();s.addEventListener('wheel',e=>{e.preventDefault();const f=Math.exp(e.deltaY*.001),b=state.box;const w=Math.min(d.width,Math.max(30,b[2]*f)),h=w*b[3]/b[2];state.box=[b[0]+(b[2]-w)/2,b[1]+(b[3]-h)/2,w,h];box();},{passive:false});s.onpointerdown=e=>{state.touches.set(e.pointerId,[e.clientX,e.clientY]);state.drag=[e.clientX,e.clientY,...state.box];s.setPointerCapture(e.pointerId);};s.onpointermove=e=>{if(!state.touches.has(e.pointerId))return;const prev=state.touches.get(e.pointerId);if(state.touches.size===2){const other=[...state.touches].find(([id])=>id!==e.pointerId)[1];const oldDist=Math.hypot(prev[0]-other[0],prev[1]-other[1]),newDist=Math.hypot(e.clientX-other[0],e.clientY-other[1]);if(oldDist>0&&newDist>0){const b=state.box,w=Math.min(d.width,Math.max(30,b[2]*oldDist/newDist)),h=w*b[3]/b[2];state.box=[b[0]+(b[2]-w)/2,b[1]+(b[3]-h)/2,w,h];}}else if(state.drag){const b=s.getBoundingClientRect(),a=state.drag;state.box=[a[2]-(e.clientX-a[0])*a[4]/b.width,a[3]-(e.clientY-a[1])*a[5]/b.height,a[4],a[5]];}state.touches.set(e.pointerId,[e.clientX,e.clientY]);box();};s.onpointerup=s.onpointercancel=e=>{state.touches.delete(e.pointerId);state.drag=null;};return s;}
for(const d of D.cards){const card=document.createElement('section');card.className='card';card.innerHTML=`<div class="heading"><h2>Obs ${String(d.index).padStart(4,'0')} · ${E(d.title)}</h2><p>${E(d.operator_note)}</p><div class="small">${E(d.captured_local)} · ${d.exposure_ms.toFixed(3)} ms / ${d.gain_db.toFixed(1)} dB · endpoint ${d.position_error_mm.toFixed(3)} mm, ${(d.orientation_error_rad*180/Math.PI).toFixed(3)}°</div><div class="counts"></div></div>`;const views=document.createElement('div');views.className='views';for(const after of[false,true]){const view=document.createElement('div');view.className='view';view.innerHTML=`<div class="caption">${after?'After · conservative optical selection':'Before · original accepted pixels'}</div>`;view.append(photo(d,after));views.append(view);}card.append(views);const detail=document.createElement('div');detail.className='detail';detail.innerHTML=`<div class="gap"><strong>Physical gap:</strong> ${d.gap?'Coupled seam and gap decisions are rejected; details below. No measured seam is asserted.':'Not qualified. Physical cut lips and a gap center have not been independently resolved in this photograph.'}</div><details><summary>Exact source and decision details</summary><p class="small">Native PNG SHA256 <code>${E(d.png_sha256)}</code><br>Source ${E(d.source_id)} · generation ${d.generation} · sequence ${d.sequence}<br>Acquisition <code>${E(d.acquisition_session_id)}</code><br>Calibration <code>${E(d.calibration_id)}</code></p><pre></pre></details>`;card.append(detail);d.render=()=>{const c=d.decisions[scope],before=scope==='saved_roi'?d.roi_points:d.context_points;card.querySelector('.counts').textContent=`${scope==='saved_roi'?'Saved ROI':'Full frame'}: before ${before?.length??'unavailable'} accepted → after ${c.selected.length} selected / ${c.rejected.length} rejected / ${c.ambiguous.length} ambiguous · ${c.status}${c.summary.overflow?' · CANDIDATE OVERFLOW, NOT QUALIFIED':''}${c.status==='ambiguous'?' · BLOCKED: cyan returns retained for provenance, not qualified for a seam':''}`;detail.querySelector('pre').textContent=JSON.stringify({scope,decision_status:c.status,physical_primary_proven:false,summary:c.summary,rejected_reason_counts:c.rejected_reason_counts,unclassified_displayed_ambiguous:c.unclassified_displayed_ambiguous,gap_inference:d.gap},null,2);};d.render();document.querySelector('#cards').append(card);}
document.querySelector('#scope').onchange=e=>{scope=e.target.value;live.forEach(s=>s.paint());D.cards.forEach(d=>d.render());};document.querySelector('#zoom').onclick=e=>{focus=!focus;live.forEach(s=>s.reset());e.target.textContent=focus?'Show full photograph':'Zoom to saved ROI';};document.querySelector('#raw').onclick=e=>{document.body.classList.toggle('raw');e.target.textContent=document.body.classList.contains('raw')?'Show dots':'Hide dots';};document.querySelector('#gamma').oninput=e=>{document.querySelector('#gv').textContent=Number(e.target.value).toFixed(2);document.querySelectorAll('feFuncR,feFuncG,feFuncB').forEach(n=>n.setAttribute('exponent',Number(e.target.value)/.5));};
})().catch(e=>{document.querySelector('#cards').textContent='Unable to decode the self-contained evidence gallery: '+e.message;});</script></html>'''


def build(args):
    decision_bytes = args.decisions.read_bytes()
    decisions = json.loads(decision_bytes)
    require(isinstance(decisions, list), 'Expected decision records list')
    by_key = {}
    normalized = []
    for item in decisions:
        if 'scope' in item:
            normalized.append(item)
            continue
        require('full_profile' in item['cloud'], 'Final pipeline evidence lacks full-frame context')
        for scope, cloud in (('saved_roi', item['cloud']), ('full_frame', item['cloud']['full_profile'])):
            normalized.append({**item, 'scope': scope, 'cloud': cloud,
                               'optical_intrinsics_id': cloud['optical_intrinsics_id']})
    for item in normalized:
        key = (item['observation'], item['scope'])
        require(key not in by_key, 'Duplicate observation/scope')
        require(key[1] in ('saved_roi', 'full_frame'), 'Unknown candidate scope')
        by_key[key] = item
    cal = helpers.read(args.calibration)
    gap = helpers.read(args.gaps) if args.gaps else None
    cards = []
    for index in ORDER:
        card = helpers.load_capture(args.captures, index, args.old_session, cal)
        snapshot = helpers.read(args.captures/f'obs-{index:04d}'/'snapshot.json')
        source_row = next(row for row in decisions if row['observation'] == f'obs-{index:04d}')
        if 'native_pixels_sha256' in source_row:
            require(source_row['native_pixels_sha256'] == snapshot['image']['native_pixels_sha256'],
                    'Decision decoded pixel hash mismatch')
        card.update(title=TITLES[index], operator_note=NOTES[index])
        card['decisions'] = {scope: validate_decision(by_key[(f'obs-{index:04d}', scope)], snapshot,
                                                     card['png_sha256'])
                             for scope in ('saved_roi', 'full_frame')}
        card['gap'] = gap_for(gap, index, card['png_sha256'])
        cards.append(card)
    output = args.output or HERE/'optical-comparison.html'
    for width, quality in ((1100, 82), (1000, 78), (900, 75), (800, 72)):
        payload_cards = []
        for card in cards:
            c = {k: v for k, v in card.items() if not k.startswith('_')}
            c['jpeg'] = helpers.jpeg(card['_image'], width, quality, display_gamma=.5)
            payload_cards.append(c)
        data = {'schema': 'same-photo-optical-comparison.v1', 'prototype': not args.final,
                'decision_file_sha256': hashlib.sha256(decision_bytes).hexdigest(),
                'display_gamma_before_jpeg': .5, 'cards': payload_cards}
        raw = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
        payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        html = HTML.replace('PAYLOAD', payload)
        if len(html.encode()) < 1_000_000:
            break
    require(len(html.encode()) < 1_000_000, 'Self-contained gallery exceeds 1MB budget')
    output.write_text(html)
    report = {'schema': 'same-photo-optical-gallery-build.v1', 'html': str(output),
              'html_bytes': len(html.encode()), 'jpeg_width': width, 'jpeg_quality': quality,
              'prototype': not args.final, 'decision_file_sha256': data['decision_file_sha256'],
              'calibration_id': helpers.canonical_hash(cal), 'all_source_checks_passed': True,
              'physical_primary_proven': False, 'counts_as_measured_seam': False,
              'observations': [{'observation': c['index'], 'native_png_sha256': c['png_sha256'],
                 'scope_counts': {s: {k: len(d[k]) for k in ('selected', 'rejected', 'ambiguous')}
                                  for s, d in c['decisions'].items()}} for c in cards]}
    output.with_suffix('.build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, default=HERE/'final-native-optical-decisions.json')
    parser.add_argument('--captures', type=Path, default=CAPTURES)
    parser.add_argument('--old-session', type=Path,
                        default=PROJECT/'runtime/agents-live/weld-paths/20261030-170257-ba242ade')
    parser.add_argument('--calibration', type=Path,
                        default=PROJECT/'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json')
    parser.add_argument('--gaps', type=Path, help='Optional per-observation, native-hash-bound gap diagnostics')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--final', action='store_true', help='Only after detector decisions have been finalized')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
