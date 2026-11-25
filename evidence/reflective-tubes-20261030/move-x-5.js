async page => {
 const target={"name": "x-5", "position_m": [-0.011144314149081909, -0.22107587054557543, 0.20544609676563177], "rpy_rad": [1.58505445553544, 0.4330801551327732, -0.3533719818143981], "planned_camera_pose": [[-0.0948377895039699, -0.9649120573869586, -0.24484794299987028, 0.024032016797331066], [0.019931585433515354, 0.24406673386428215, -0.9695535886803414, -0.22133506448855406], [0.9952931857418233, -0.09683052685020738, -0.003914522423504939, 0.21528858323674377], [0.0, 0.0, 0.0, 1.0]], "purpose": "bounded optical diagnosis; predicted pixel/plane pivot is not a measured seam"};
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
