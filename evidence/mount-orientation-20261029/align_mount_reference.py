"""One-off CAD-only alignment; hardware, nominal motion TCP and hand-eye stay fixed."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

root=Path('/home/faraz/Code/autoweld')
out=root/'evidence/mount-orientation-20261029'
response=json.loads((out/'rest-capture.json').read_text())
proof=response['result']
q=np.array(proof['joint_rad'])
m=json.loads((out/'manifest-before.json').read_text())
u=yourdfpy.URDF.load(str(out/'mount-before.urdf'),load_meshes=False,build_scene_graph=True)
u.update_cfg(dict(zip([f'joint{i+1}' for i in range(6)],q)))
f=u.get_transform('gripper')
def mesh_center(index):
 e=m['meshes'][index]
 mesh=trimesh.load(root/'assets/camera_mount'/e['file'],process=False)
 local=mesh.bounds.mean(axis=0)*np.array(e['scale'])
 return Rotation.from_euler('xyz',e['rpy']).apply(local)+e['xyz']
camera=mesh_center(2)  # FLIR camera case optical_head_1
laser=mesh_center(22)  # Laser module optical_head_21
stack=f[:3,:3]@(camera-laser);stack/=np.linalg.norm(stack)
forward=f[:3,:3]@np.array(m['flange_from_camera'])[:3,2]
side=forward-stack*(forward@stack);side/=np.linalg.norm(side)
up=np.array([0.,0.,1.])
heading=forward.copy();heading[2]=0.;heading/=np.linalg.norm(heading)
world_rotation=np.column_stack([up,heading,np.cross(up,heading)]) @ np.column_stack([stack,side,np.cross(stack,side)]).T
rotation=f[:3,:3].T@world_rotation@f[:3,:3]
c=np.eye(4);c[:3,:3]=rotation
for group in ('meshes','envelopes'):
 for e in m[group]:
  e['xyz']=(rotation@np.array(e['xyz'])).tolist()
  e['rpy']=Rotation.from_matrix(rotation@Rotation.from_euler('xyz',e['rpy']).as_matrix()).as_euler('xyz').tolist()
for name,value in list(m['frames'].items()):
 if name != 'active_tcp':m['frames'][name]=(c@np.array(value)).tolist()
m['flange_from_camera']=m['frames']['camera_optical']
m['model_id']='autoweld-receiver-camera-sixaxis-installed-limits-v2-vertical-stack-20261029'
m['display_alignment']={
 'schema':'autoweld.mount-display-alignment.v1','basis':'operator-requested unchanged-hardware CAD correction at fresh disabled SDK pose',
 'reference_joint_rad':q.tolist(),'reference_physical_timestamp':proof['physical_timestamp'],
 'corrected_flange_from_imported_flange':c.tolist(),
 'camera_center_flange_m':(rotation@camera).tolist(),'laser_center_flange_m':(rotation@laser).tolist(),
 'center_sources':['FLIR case optical_head_1 mesh bounding-box center','laser module optical_head_21 mesh bounding-box center'],
 'stack_direction_base':[0.,0.,1.],'nominal_motion_tcp_preserved':True,'measured_handeye_preserved':True,
 'qualification':'CAD mechanical centers, not a new optical or laser calibration'}
p=root/'assets/camera_mount/manifest.json';p.write_text(json.dumps(m,indent=2)+'\n')
p=root/'assets/camera_mount/camera_mount.urdf';tree=ET.parse(p)
for i,e in enumerate(m['meshes']):
 node=tree.find(f"./joint[@name='mount_{i}_fixed']/origin")
 node.set('xyz',' '.join(format(v,'.16g') for v in e['xyz']))
 node.set('rpy',' '.join(format(v,'.16g') for v in e['rpy']))
for joint,frame in [('camera_optical_fixed','camera_optical'),('weld_tcp_fixed','active_tcp')]:
 t=np.array(m['frames'][frame]);node=tree.find(f"./joint[@name='{joint}']/origin")
 node.set('xyz',' '.join(format(v,'.16g') for v in t[:3,3]))
 node.set('rpy',' '.join(format(v,'.16g') for v in Rotation.from_matrix(t[:3,:3]).as_euler('xyz')))
ET.indent(tree,space='  ');tree.write(p,encoding='utf-8',xml_declaration=True)
a=f[:3,:3]@np.array(m['display_alignment']['camera_center_flange_m'])+f[:3,3]
b=f[:3,:3]@np.array(m['display_alignment']['laser_center_flange_m'])+f[:3,3]
report=dict(alignment=m['display_alignment'],camera_center_base_m=a.tolist(),laser_center_base_m=b.tolist(),
 horizontal_center_error_mm=float(np.linalg.norm((a-b)[:2])*1000),
 camera_axis_base=(f[:3,:3]@np.array(m['flange_from_camera'])[:3,2]).tolist(),
 motion_tcp_unchanged=m['frames']['active_tcp']==json.loads((out/'manifest-before.json').read_text())['frames']['active_tcp'])
(out/'alignment-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
