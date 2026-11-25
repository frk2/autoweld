async page => {
 const target={"name": "x+5", "position_m": [-0.0349530036609023, -0.22427794106434973, 0.20324157936954118], "rpy_rad": [1.5664853489104367, 0.432012272205506, -0.18744047095910343], "planned_camera_pose": [[-0.0948377895039699, -0.9927702741970613, -0.0735729322012696, -0.00024151812954577995], [0.019931585433515354, 0.07199759793713847, -0.9972056346578618, -0.218289597473571], [0.9952931857418233, -0.09603920325561899, 0.012959392498059481, 0.21291466024022948], [0.0, 0.0, 0.0, 1.0]], "purpose": "bounded optical diagnosis; predicted pixel/plane pivot is not a measured seam"};
 const mode=page.getByLabel('Mode',{exact:true});await mode.waitFor({timeout:30000});
 if(await mode.inputValue()!=='IK'){
  await mode.click({force:true});await page.getByRole('option',{name:'IK',exact:true}).click({force:true});
  await page.getByText(/mode: accepted/).first().waitFor({timeout:10000});
 }
 for(const [label,values]of[['Position (m)',target.position_m],['Roll / pitch / yaw (rad)',target.rpy_rad]]){
  let group=page.getByLabel(label,{exact:true}).locator('..');
  while(await group.locator('input').count()<3)group=group.locator('..');
  for(let i=0;i<3;i++){
   const input=group.locator('input').nth(i);await input.fill(String(values[i]));await input.press('Tab');
  }
 }
 await page.getByRole('button',{name:'Send TCP pose',exact:true}).click({force:true});
 return {submitted:target.name,position_m:target.position_m,rpy_rad:target.rpy_rad};
}
