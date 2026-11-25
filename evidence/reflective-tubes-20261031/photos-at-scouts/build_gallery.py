#!/usr/bin/env python3
"""Build an offline gallery from existing native recaptures; never acquire or detect."""
import argparse
import base64
import hashlib
import io
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

for thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'

import numpy as np  # noqa: E402 -- bound numerical workers before loading NumPy.
from PIL import Image, ImageDraw  # noqa: E402

OBSERVATIONS = (6, 8, 15, 47, 53)
TITLES = {6: 'Center · 0° tilt / 0 mm offset', 8: 'First failing view · +5° / +3 mm',
          15: 'Opposite view · −5° / −3 mm', 47: 'Positive edge · +10° / +9 mm',
          53: 'Usable negative edge · −10° / −9 mm'}
IDENTITY = ('source_id', 'generation', 'sequence', 'acquisition_session_id', 'calibration_id')
BUDGET = 1_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def finite(value, shape, label):
    result = np.asarray(value, float)
    require(result.shape == shape and np.isfinite(result).all(), f'{label}: invalid shape or nonfinite data')
    return result


def rigid(value, label):
    p = finite(value, (4, 4), label)
    require(np.allclose(p[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            and np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            and abs(np.linalg.det(p[:3, :3])-1) < 1e-6, f'{label}: not a rigid pose')
    return p


def status_pose(status):
    q = finite(status['tcp_wxyz'], (4,), 'Measured TCP quaternion')
    require(abs(np.linalg.norm(q)-1) < 1e-5, 'Measured TCP quaternion is not normalized')
    w, x, y, z = q/np.linalg.norm(q)
    p = np.eye(4)
    p[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    p[:3, 3] = finite(status['tcp_position_m'], (3,), 'Measured TCP position')
    return p


def pose_error(actual, target):
    distance = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(actual[:3, :3] @ target[:3, :3].T)-1)/2, -1, 1)))
    return distance, angle


def uv_for(cloud, metadata, size, label):
    if cloud is None:
        return None
    require(all(cloud.get(key) == metadata.get(key) for key in IDENTITY), f'{label}: points belong to another exposure')
    require(cloud.get('controls') == metadata.get('controls') and cloud.get('read_started') == metadata.get('read_started'),
            f'{label}: native controls/acquisition time differ from the photo')
    if cloud.get('optical_intrinsics_id') is not None:
        require(cloud['optical_intrinsics_id'] == metadata.get('optical_intrinsics_id'), f'{label}: optical calibration differs')
    require(cloud.get('units') == 'mm' and cloud.get('coordinate_frame') == 'camera', f'{label}: unexpected native frame/units')
    uv, xyz = np.asarray(cloud.get('uv'), float), np.asarray(cloud.get('xyz_mm'), float)
    if uv.shape == xyz.shape == (0,):
        uv, xyz = uv.reshape(0, 2), xyz.reshape(0, 3)
    require(uv.shape == (len(xyz), 2) and xyz.shape == (len(uv), 3)
            and np.isfinite(uv).all() and np.isfinite(xyz).all()
            and type(cloud.get('count')) is int and cloud['count'] == len(uv), f'{label}: malformed native arrays/count')
    require(np.all(uv >= 0) and np.all(uv < np.asarray(size)), f'{label}: points outside native image')
    # Rounding is confined to display geometry; original files remain unchanged.
    return np.round(uv, 4).tolist()


def load_capture(root, index, old_session, calibration, old_profiles=None):
    directory = root/f'obs-{index:04d}'
    png = (directory/'native.png').read_bytes()
    snapshot, station = read(directory/'snapshot.json'), read(directory/'station.json')
    require(snapshot.get('schema') == 'weldfinder.native_laser_snapshot.v1', 'Unsupported native snapshot')
    metadata, info = snapshot['metadata'], snapshot['image']
    require(info.get('encoding') == 'png' and digest(png) == info.get('sha256'), f'{directory}: PNG SHA256 differs')
    require(all(info.get(key) is False for key in ('annotated', 'scaled', 'tone_mapped')), 'Snapshot is not original native pixels')
    require(info.get('dtype') == 'uint8', 'Only native uint8 camera exports are supported')
    image = Image.open(io.BytesIO(png))
    pixels = np.asarray(image)
    if info.get('channel_order') == 'BGR':
        require(pixels.ndim == 3 and pixels.shape[2] == 3, 'Malformed color image')
        native_pixels = np.ascontiguousarray(pixels[..., ::-1])
    else:
        require(info.get('channel_order') == 'mono' and pixels.ndim == 2, 'Unexpected native channel order')
        native_pixels = pixels
    require(digest(native_pixels.tobytes()) == info.get('native_pixels_sha256'), 'Decoded native pixel hash differs')
    size = list(image.size)
    require(size == info.get('image_size') == calibration.get('image_size'), 'Native/calibration image resolution differs')
    calibration_id = canonical_hash(calibration)
    require(metadata.get('calibration_id') == calibration_id and metadata.get('source_id') == calibration.get('source_id'),
            'Snapshot does not match the supplied live calibration')
    optical = {key: calibration[key] for key in ('schema_version', 'source_id', 'image_size',
               'camera_matrix', 'distortion', 'board', 'synthetic')}
    require(metadata.get('optical_intrinsics_id') == canonical_hash(optical), 'Optical calibration identity differs')
    require(metadata.get('controls_settled') is True, 'Camera controls had not settled for this exposure')
    require(all(station.get(key) is True for key in ('recapture', 'not_same_acquisition_as_old_profile',
                'not_new_seam_measurement', 'camera_pose_comparison_only')), 'Recapture scope flags are missing')
    require(station.get('old_observation') == index, 'Station refers to another archived observation')
    old_path = old_session/f'observation-{index:04d}.json'
    old_bytes = old_path.read_bytes()
    require(digest(old_bytes) == station.get('old_native_sha256'), 'Archived profile SHA256 differs')
    old = json.loads(old_bytes)
    require(old['measurement']['cloud']['calibration_id'] == calibration_id, 'Archived/current calibrations differ')
    old_camera = rigid(station['old_base_from_camera'], 'Archived camera pose')
    require(np.allclose(old_camera, rigid(old['measurement']['base_from_camera'], 'Native archived camera'), atol=1e-12, rtol=0),
            'Station substituted the archived measured camera pose')
    target = rigid(station['requested_base_from_tcp'], 'Requested recapture TCP')
    require(any(np.allclose(target, rigid(old['measurement'][key], key), atol=1e-12, rtol=0)
                for key in ('base_from_tcp', 'requested_base_from_tcp')), 'Target differs from the archived measured/requested TCP')
    for label in ('before', 'after'):
        status = station[label]
        require(status.get('closed') is False and status.get('motion_active') is False and not status.get('fault')
                and (status.get('sdk') or {}).get('feedback_fresh') is True,
                f'{label}: fresh stationary healthy motor feedback is absent')
    error_mm, error_rad = pose_error(status_pose(station['after']), target)
    error_mm *= 1000
    require(abs(error_mm-station['endpoint_position_error_mm']) < 1e-6
            and abs(error_rad-station['endpoint_orientation_error_rad']) < 1e-6, 'Reported endpoint errors disagree with measured pose')
    require(error_mm <= .5 and error_rad <= .005, 'Recapture did not converge to the requested TCP endpoint')
    roi = finite(snapshot['measurement_settings']['detection']['roi'], (4,), 'Native detection ROI')
    require(np.all((roi >= 0) & (roi <= 1)) and np.all(roi[2:] > roi[:2]), 'Invalid normalized detection ROI')
    require(np.allclose(roi, finite(station['roi'], (4,), 'Requested ROI'), atol=1e-12, rtol=0), 'Station/native detection ROI differs')
    cloud = snapshot.get('cloud')
    points = uv_for(cloud, metadata, size, 'ROI returns')
    full = cloud.get('full_profile') if cloud is not None else None
    context = uv_for(full, metadata, size, 'Full-context returns')
    capture = datetime.fromisoformat(station['capture_utc'].replace('Z', '+00:00'))
    require(capture.tzinfo is not None, 'Capture UTC is timezone-free')
    local = capture.astimezone(ZoneInfo('America/Los_Angeles'))
    old_profile = None
    candidates = [directory/'old-profile.png']
    if old_profiles is not None:
        candidates += [old_profiles/f'obs-{index:04d}.png', old_profiles/f'observation-{index:04d}.png']
    for candidate in candidates:
        if candidate.is_file():
            old_profile = Image.open(candidate).convert('RGB').copy()
            break
    controls = metadata['controls']
    crop = (max(0, math.floor(roi[0]*size[0])-12), max(0, math.floor(roi[1]*size[1])-12),
            min(size[0], math.ceil(roi[2]*size[0])+12), min(size[1], math.ceil(roi[3]*size[1])+12))
    return dict(index=index, title=TITLES[index], width=size[0], height=size[1],
        captured_local=local.strftime('%b %d, %Y · %H:%M:%S %Z'), captured_utc=station['capture_utc'],
        exposure_ms=float(controls['exposure_us'])/1000, gain_db=float(controls['gain_db']),
        position_error_mm=error_mm, orientation_error_rad=error_rad,
        roi=roi.tolist(), crop=list(crop), roi_points=points, context_points=context,
        cloud_error=snapshot.get('cloud_error'), old_status=old['result'].get('status'),
        old_reason=(old['result'].get('quality') or {}).get('reason'),
        old_roi_count=old['measurement']['cloud'].get('count'),
        source_id=metadata['source_id'], sequence=metadata['sequence'], generation=metadata['generation'],
        acquisition_session_id=metadata['acquisition_session_id'], calibration_id=calibration_id,
        png_sha256=info['sha256'], old_native_sha256=station['old_native_sha256'],
        _image=image.convert('RGB').copy(), _old_profile=old_profile)


def jpeg(image, width, quality, *, display_gamma=None):
    image = image.copy()
    if display_gamma is not None:
        # Brighten the native display copy BEFORE lossy encoding; otherwise a
        # browser gamma lift magnifies quantization blocks in dark JPEG pixels.
        lookup = [round(255*(value/255)**display_gamma) for value in range(256)]
        image = image.point(lookup*3)
    if image.width > width:
        image.thumbnail((width, 100000), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=quality, optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def export_marked_displays(card, directory):
    """Display-only copies: original PNG and all native measurements stay intact."""
    lookup = [round(255*(value/255)**.5) for value in range(256)]
    display = card['_image'].point(lookup*3)
    display.save(directory/'unmarked-display.jpg', quality=94, optimize=True)
    draw = ImageDraw.Draw(display)
    for points, color in ((card['context_points'], '#32d0ef'), (card['roi_points'], '#41ff76')):
        for u, v in points or []:
            draw.ellipse((u-2.1, v-2.1, u+2.1, v+2.1), fill=color, outline='#060d10', width=1)
    roi, w, h = card['roi'], card['width'], card['height']
    draw.rectangle((roi[0]*w, roi[1]*h, roi[2]*w, roi[3]*h), outline='#ffc65d', width=2)
    display.save(directory/'marked-display.jpg', quality=94, optimize=True)
    display.crop(tuple(card['crop'])).save(directory/'roi-marked-display.jpg', quality=95, optimize=True)
    return dict(marked=str(directory/'marked-display.jpg'), crop=str(directory/'roi-marked-display.jpg'),
                unmarked=str(directory/'unmarked-display.jpg'), gamma=.5, native_measurements_unchanged=True)


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Laser pictures at the five scout poses</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101722;color:#edf2f8;font:15px/1.45 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:24px}h1{font-size:clamp(24px,4vw,35px);line-height:1.12;margin:0 0 12px}h2{font-size:19px;margin:0 0 5px}p{margin:8px 0;color:#c3cfdf}.notice{border-left:4px solid #e6bb57;background:#222a35;padding:12px 16px;margin:18px 0}.controls{position:sticky;top:0;z-index:3;display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#101722ed;padding:12px 0;border-bottom:1px solid #344154}button{background:#294760;color:#fff;border:1px solid #5781a0;border-radius:6px;padding:8px 13px;font:inherit;cursor:pointer}button[aria-pressed=true]{background:#326f52;border-color:#69ba89}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:#bfcddd}.green{color:#59ff88}.cyan{color:#51d5f1}.amber{color:#ffc65d}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,490px),1fr));gap:20px;margin-top:18px}.card{border:1px solid #344154;border-radius:10px;background:#182230;overflow:hidden}.heading{padding:16px 16px 10px}.small{font-size:12px;color:#9fb2ca}.badges{display:flex;gap:8px;flex-wrap:wrap;font-size:12px;margin-top:10px}.badges span{background:#283546;border-radius:4px;padding:4px 7px}svg.photo{display:block;width:100%;height:auto;background:#05080b;touch-action:none}.crop-block{border-top:1px solid #344154}.crop-title{padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}.crop-title input{max-width:150px}.crop{aspect-ratio:2.1;width:100%;cursor:grab}.crop:active{cursor:grabbing}.raw .marks{display:none}.hide-context .context-points{display:none}.detail{padding:12px 16px}.old-profile{width:100%;height:auto;display:block;background:#fff}.pending{padding:35px;color:#a7b8cd}details{font-size:12px;color:#a7b8cd}code{font-size:11px;overflow-wrap:anywhere}summary{cursor:pointer}.footer{margin-top:20px;padding:14px;border-top:1px solid #344154;font-size:12px;color:#9bb0c9}@media(max-width:600px){main{padding:16px 10px}.heading{padding:13px 12px}.controls{gap:8px}.controls label{font-size:13px}.crop-title{padding:10px 12px}}</style>
<main><h1>Camera pictures at the five scout poses</h1><p>Fresh recaptures, with the laser-return pixels marked on their own source images.</p>
<div class="notice"><strong>These are new photographs, not the original October 30 exposures.</strong> The arm returned to the archived scout poses for these recaptures. The displayed dots are this new exposure’s measured laser pixels. These pictures are for comparison; they are not new weld-seam or angular-coverage measurements.</div>
<div class="controls"><button id="marked" aria-pressed="true">Marked</button><button id="raw" aria-pressed="false">Unmarked / raw display</button><label>Display gamma <input id="gamma" type="range" min="0.2" max="1" step="0.05" value="0.5"> <output id="gamma-value">0.5</output></label><label><input id="context" type="checkbox" checked> Full context</label></div>
<p class="legend"><span class="green">● Accepted ROI returns</span><span class="cyan">● Accepted full-context returns, if captured</span><span class="amber">▭ Actual detection ROI</span></p><p class="small">Gamma brightens display pixels only. No re-detection, smoothing, line extension or gap filling. The close-up uses a native-resolution crop. Drag to pan; use the slider to zoom.</p>
<svg width="0" height="0" aria-hidden="true"><defs><filter id="display-gamma" x="0" y="0" width="100%" height="100%" color-interpolation-filters="sRGB"><feComponentTransfer><feFuncR type="gamma" amplitude="1" exponent="1"/><feFuncG type="gamma" amplitude="1" exponent="1"/><feFuncB type="gamma" amplitude="1" exponent="1"/></feComponentTransfer></filter></defs></svg>
<div class="grid" id="cards"></div><p class="footer">PNG byte hashes, decoded native pixel hashes, source/frame identity and the supplied live calibration identity were verified before rendering. Endpoint errors come from fresh SDK pose feedback. Display JPEGs are reduced/compressed; the original PNGs and native measurements remain unchanged. The marked/unmarked toggle changes display only.</p></main>
<script id="gallery-data" type="application/json">PAYLOAD</script><script>
const DATA=JSON.parse(document.querySelector('#gallery-data').textContent),NS='http://www.w3.org/2000/svg';
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function node(name,attrs){const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,v);return n;}
function marks(d){const g=node('g',{class:'marks'});for(const [kind,points,color]of[['context-points',d.context_points,'#32d0ef'],['roi-points',d.roi_points,'#41ff76']]){const group=node('g',{class:kind,fill:color,stroke:'#060d10','stroke-width':.7});for(const p of points||[])group.append(node('circle',{cx:p[0],cy:p[1],r:2.1}));g.append(group);}g.append(node('rect',{x:d.roi[0]*d.width,y:d.roi[1]*d.height,width:(d.roi[2]-d.roi[0])*d.width,height:(d.roi[3]-d.roi[1])*d.height,fill:'none',stroke:'#ffc65d','stroke-width':1.8,'stroke-dasharray':'7 5'}));return g;}
function photo(d,crop){const box=crop?d.crop:[0,0,d.width,d.height],w=box[2]-box[0],h=box[3]-box[1],s=node('svg',{viewBox:`${box[0]} ${box[1]} ${w} ${h}`,class:'photo'+(crop?' crop':''),role:'img','aria-label':`${crop?'Laser ROI close-up':'Camera photograph'} for old observation ${d.index}, fresh ${d.captured_utc}`});s.append(node('image',{href:crop?d.crop_jpeg:d.jpeg,x:box[0],y:box[1],width:w,height:h,filter:'url(#display-gamma)','preserveAspectRatio':'none'}));s.append(marks(d));return s;}
for(const d of DATA.cards){const card=document.createElement('section');card.className='card';if(d.pending){card.innerHTML=`<div class="heading"><h2>Obs ${String(d.index).padStart(4,'0')} · ${esc(d.title)}</h2></div><div class="pending">Recapture not yet available. No image or points are invented.</div>`;document.querySelector('#cards').append(card);continue;}
const rn=d.roi_points===null?'Unavailable':d.roi_points.length,cn=d.context_points===null?'Not captured':d.context_points.length;
card.innerHTML=`<div class="heading"><h2>Obs ${String(d.index).padStart(4,'0')} · ${esc(d.title)}</h2><div class="small">Fresh recapture: ${esc(d.captured_local)}</div><div class="badges"><span>${d.exposure_ms.toFixed(3)} ms · ${d.gain_db.toFixed(1)} dB</span><span class="green">ROI ${rn}</span><span class="cyan">Context ${cn}</span></div><p class="small">TCP endpoint error: ${d.position_error_mm.toFixed(3)} mm · ${(d.orientation_error_rad*180/Math.PI).toFixed(3)}°</p></div>`;
card.append(photo(d,false));const block=document.createElement('div');block.className='crop-block';block.innerHTML='<div class="crop-title"><strong>Native ROI close-up</strong><label>Zoom <input type="range" min="1" max="4" step="0.1" value="1"></label></div>';
const crop=photo(d,true);block.append(crop);card.append(block);const x0=d.crop[0],y0=d.crop[1],W=d.crop[2]-x0,H=d.crop[3]-y0;let zoom=1,cx=x0+W/2,cy=y0+H/2,drag=null;
const update=()=>{const w=W/zoom,h=H/zoom;cx=Math.max(x0+w/2,Math.min(x0+W-w/2,cx));cy=Math.max(y0+h/2,Math.min(y0+H-h/2,cy));crop.setAttribute('viewBox',`${cx-w/2} ${cy-h/2} ${w} ${h}`);};
block.querySelector('input').addEventListener('input',e=>{zoom=Number(e.target.value);update();});crop.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,cx,cy];crop.setPointerCapture(e.pointerId);});crop.addEventListener('pointermove',e=>{if(!drag)return;const b=crop.getBoundingClientRect();cx=drag[2]-(e.clientX-drag[0])*W/(zoom*b.width);cy=drag[3]-(e.clientY-drag[1])*H/(zoom*b.height);update();});crop.addEventListener('pointerup',()=>drag=null);crop.addEventListener('pointercancel',()=>drag=null);
const detail=document.createElement('div');detail.className='detail';detail.innerHTML=`<p><strong>October 30 archived profile:</strong> ${esc(d.old_status)} · ${d.old_roi_count} ROI points.<br><span class="small">${esc(d.old_reason)}</span></p><details><summary>Source and integrity details</summary><p>Source: ${esc(d.source_id)}<br>Frame ${d.sequence} · generation ${d.generation}<br>Acquisition: <code>${esc(d.acquisition_session_id)}</code><br>Native PNG SHA256: <code>${d.png_sha256}</code><br>Calibration: <code>${d.calibration_id}</code><br>Archived native acquisition SHA256: <code>${d.old_native_sha256}</code></p></details>`;card.append(detail);
if(d.old_profile_jpeg){const label=document.createElement('p');label.className='detail small';label.textContent='October 30 profile plot — recorded UV/XYZ, not a camera photograph.';card.append(label);const image=document.createElement('img');image.className='old-profile';image.alt=`Archived profile plot for observation ${d.index}`;image.src=d.old_profile_jpeg;card.append(image);}document.querySelector('#cards').append(card);}
for(const id of ['marked','raw'])document.getElementById(id).onclick=()=>{document.body.classList.toggle('raw',id==='raw');document.getElementById('raw').setAttribute('aria-pressed',id==='raw');document.getElementById('marked').setAttribute('aria-pressed',id==='marked');};
document.querySelector('#context').onchange=e=>document.body.classList.toggle('hide-context',!e.target.checked);document.querySelector('#gamma').oninput=e=>{document.querySelector('#gamma-value').textContent=Number(e.target.value).toFixed(2);document.querySelectorAll('feFuncR,feFuncG,feFuncB').forEach(n=>n.setAttribute('exponent',Number(e.target.value)/0.5));};
</script></html>'''


def build(args):
    project = Path(__file__).resolve().parents[3]
    calibration_path = args.calibration or project/'runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json'
    old_session = args.old_session or project/'runtime/agents-live/weld-paths/20261030-170257-ba242ade'
    calibration = read(calibration_path)
    cards = []
    for index in OBSERVATIONS:
        folder = args.capture_root/f'obs-{index:04d}'
        ready = all((folder/name).is_file() for name in ('native.png', 'snapshot.json', 'station.json'))
        if not ready:
            require(args.allow_partial, f'{folder}: native.png, snapshot.json or station.json is missing')
            cards.append(dict(index=index, title=TITLES[index], pending=True))
        else:
            cards.append(load_capture(args.capture_root, index, old_session, calibration, args.old_profiles))
    exports = {f'obs-{card["index"]:04d}': export_marked_displays(card, args.capture_root/f'obs-{card["index"]:04d}')
               for card in cards if not card.get('pending')}
    for width, quality in ((960, 84), (900, 80), (800, 78), (720, 76), (640, 72), (560, 68)):
        display = []
        for card in cards:
            item = {key: value for key, value in card.items() if not key.startswith('_')}
            if not card.get('pending'):
                item['jpeg'] = jpeg(card['_image'], width, quality, display_gamma=.5)
                item['crop_jpeg'] = jpeg(card['_image'].crop(tuple(card['crop'])), 600, max(quality, 85), display_gamma=.5)
                if card['_old_profile'] is not None:
                    item['old_profile_jpeg'] = jpeg(card['_old_profile'], 650, quality)
            display.append(item)
        payload = json.dumps(dict(cards=display), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
        rendered = TEMPLATE.replace('PAYLOAD', payload).encode()
        if len(rendered) < BUDGET:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(rendered)
            return dict(status='ready', output=str(args.output), html_bytes=len(rendered),
                photo_display_max_width=width, jpeg_quality=quality,
                verified_recaptures=sum(not c.get('pending', False) for c in cards),
                expected_recaptures=len(OBSERVATIONS), display_gamma_only=True,
                display_exports=exports,
                overlay_source='Exact same-snapshot accepted native UV; no re-detection or gap filling',
                publication_performed=False)
    raise ValueError('Gallery exceeds 1 MB; remove optional profile plots or use a smaller explicit display budget')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'gallery.html')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--old-session', type=Path)
    parser.add_argument('--old-profiles', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    print(json.dumps(build(parser.parse_args()), indent=2))
