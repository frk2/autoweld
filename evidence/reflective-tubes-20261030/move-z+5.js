async page => {
 const target={"name": "z+5", "position_m": [-0.022506751387446775, -0.2222696426294086, 0.20089879429744942], "rpy_rad": [1.5859333595477456, 0.3454362033310121, -0.26615458548369675], "planned_camera_pose": [[-0.18011440848189614, -0.9705755078072453, -0.15981859560740938, 0.011602855267074129], [0.03368176272219091, 0.15629501376811386, -0.9871359620291177, -0.2192240294631547], [0.9830688372220651, -0.18318038190792468, 0.004539710004420013, 0.21361000803182878], [0.0, 0.0, 0.0, 1.0]], "purpose": "bounded optical diagnosis; predicted pixel/plane pivot is not a measured seam"};
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
