async page => {
 await page.getByLabel('Mode',{exact:true}).waitFor({timeout:30000});
 const panel=page.getByLabel('Calibration panel',{exact:true});await panel.click({force:true});
 await page.getByRole('option',{name:'Weld seam finding',exact:true}).click({force:true});
 const pose=page.getByLabel('Weld start pose',{exact:true});await pose.click({force:true});
 await page.getByRole('option',{name:'Weld Finder 2 glare-safe',exact:true}).click({force:true});
 const shiny=page.getByLabel('Reflective surface (low gain and glare rejection)',{exact:true});if(!await shiny.isChecked())await shiny.check({force:true});
 const roi=page.getByLabel('Restrict seam search region',{exact:true});
 if(!await roi.isVisible())await page.getByText('Seam search region (optional)',{exact:true}).click({force:true});
 if(!await roi.isChecked())await roi.check({force:true});
 const settings={'Seam ROI left (%)':300/1440*100,'Seam ROI top (%)':140/1080*100,'Seam ROI right (%)':650/1440*100,'Seam ROI bottom (%)':28};
 for(const [label,value]of Object.entries(settings)){const input=page.getByLabel(label,{exact:true});await input.fill(String(value));await input.press('Tab');}
 const enable=page.getByRole('button',{name:'Enable motors',exact:true});if(await enable.isEnabled())await enable.click({force:true});
 const start=page.getByRole('button',{name:'Find weld seam',exact:true});await start.waitFor({timeout:15000});
 await page.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Find weld seam'&&!b.disabled),null,{timeout:20000});
 const actual={};for(const label of Object.keys(settings))actual[label]=await page.getByLabel(label,{exact:true}).inputValue();
 await start.click({force:true});
 return {submitted:'one service-owned coped-tube weld_start',pose:await pose.inputValue(),reflective:await shiny.isChecked(),regionPercent:actual};
}
