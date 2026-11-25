async page => {
 const target={"name": "y-5", "position_m": [-0.02427229422784295, -0.22228985492016146, 0.2158296990523378], "rpy_rad": [1.6712631065459058, 0.4396574128969637, -0.22160802652210432], "planned_camera_pose": [[-0.10840601148764717, -0.9825801800260763, -0.15094477961580538, 0.010645309545537673], [-0.06617882823014118, 0.15863582310107435, -0.985116763801698, -0.21998762901059457], [0.9919014564801415, -0.09680323057092996, -0.08222308181773423, 0.2263071358988277], [0.0, 0.0, 0.0, 1.0]], "purpose": "bounded optical diagnosis; predicted pixel/plane pivot is not a measured seam"};
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
