async page => {
 const target={"name": "y+5", "position_m": [-0.02212454890618334, -0.22301554464809373, 0.1928402558386388], "rpy_rad": [1.4812468361693873, 0.4205724550430423, -0.31797916504353196], "planned_camera_pose": [[-0.08054779467754442, -0.9825801800260763, -0.1674760955853345, 0.013111742562551185], [0.1058903076970025, 0.15863582310107435, -0.9816424595365052, -0.21963180582513447], [0.991110132885553, -0.09680323057092996, 0.09126795189228877, 0.20185524643175956], [0.0, 0.0, 0.0, 1.0]], "purpose": "bounded optical diagnosis; predicted pixel/plane pivot is not a measured seam"};
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
