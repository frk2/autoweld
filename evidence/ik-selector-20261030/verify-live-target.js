async page => {
  const expected={"tcp_position_m": [-0.02305557902162253, -0.22217355402629538, 0.20433178759170262], "tcp_wxyz": [0.6802875817896364, 0.6809011126529466, 0.09505183407093962, -0.2540622555409176], "tcp_rpy_rad": [1.5086098257067229, 0.4953137404334229, -0.24830896490236187]};
  const readVector=async label=>page.getByLabel(label,{exact:true}).evaluate(input=>{
    let group=input.parentElement;
    while(group && group.querySelectorAll('input').length<3) group=group.parentElement;
    return Array.from(group.querySelectorAll('input')).slice(0,3).map(node=>Number(node.value));
  });
  await page.waitForFunction(expected=>{
    const label=Array.from(document.querySelectorAll('label')).find(x=>x.textContent==='Position (m)');
    const input=label && document.getElementById(label.htmlFor);
    return input && Math.abs(Number(input.value)-expected.tcp_position_m[0])<.002;
  },expected,{timeout:10000});
  const position=await readVector('Position (m)');
  const rpy=await readVector('Roll / pitch / yaw (rad)');
  const maxPositionError=Math.max(...position.map((x,i)=>Math.abs(x-expected.tcp_position_m[i])));
  const maxRpyError=Math.max(...rpy.map((x,i)=>Math.abs(x-expected.tcp_rpy_rad[i])));
  if(maxPositionError>.002 || maxRpyError>.02) throw new Error(JSON.stringify({position,rpy,expected,maxPositionError,maxRpyError}));
  const panel=page.getByLabel('Calibration panel',{exact:true});
  await panel.click({force:true});
  await page.getByRole('option',{name:'Weld seam finding',exact:true}).click({force:true});
  return {result:'PASS',mode:await page.getByLabel('Mode',{exact:true}).inputValue(),position,rpy,expected,maxPositionErrorM:maxPositionError,maxRpyErrorRad:maxRpyError};
}
