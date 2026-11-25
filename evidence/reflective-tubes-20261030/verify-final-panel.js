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
 const summary=page.getByText('Local seam locator detected; coped-tube tracing is not qualified.',{exact:false}).first();
 await summary.waitFor({timeout:15000});
 return {result:'PASS',summary:await summary.innerText(),pose:await pose.inputValue(),reflective:await shiny.isChecked(),regionBottomPercent:await page.getByLabel('Seam ROI bottom (%)',{exact:true}).inputValue(),fullProfilesRed:await page.getByLabel('Show full laser profiles in red',{exact:true}).isChecked(),seamGreen:await page.getByLabel('Show weld seam in green',{exact:true}).isChecked(),replayDisabled:await page.getByRole('button',{name:'Replay weld path',exact:true}).isDisabled()};
}
