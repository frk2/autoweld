async page => {
  const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.getByLabel('Mode',{exact:true}).waitFor({timeout:30000});
  const enable=page.getByRole('button',{name:'Enable motors',exact:true});
  if(await enable.isEnabled()) await enable.click({force:true});
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Enable motors'&&b.disabled),null,{timeout:20000});
  const q=[-1.5264744029907682, 0.5323491264209963, 0.38586251621271117, 0.20542458228427485, 0.26379034103914023, -3.652819104295414];
  for(let i=0;i<6;i++){
    const input=page.getByLabel(`Joint ${i+1} (rad)`,{exact:true});
    await input.fill(String(q[i]));
    await input.press('Tab');
  }
  const move=page.getByRole('button',{name:'Send joint waypoint',exact:true});
  await move.click({force:true});
  return {submitted:'restore prior measured joints through existing panel',q,errors};
}
