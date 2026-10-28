// Read-only checks against an offline or live service; never starts camera/motors/motion.
async page => {
  // Viser applies incoming updates on rendered frames and gates reconnects on focus.
  await page.bringToFront();
  await page.setViewportSize({width: 1500, height: 1200});
  const selector = page.getByLabel('Calibration panel', {exact: true});
  await selector.waitFor();
  if (await selector.inputValue() !== 'Laser scanning') {
    if (await selector.evaluate(element => element.tagName === 'SELECT')) {
      await selector.selectOption({label: 'Laser scanning'});
    } else {
      await selector.click({force: true});
      await page.getByRole('option', {name: 'Laser scanning', exact: true}).click({force: true});
    }
  }
  await page.waitForFunction(() => [...document.querySelectorAll('button')]
    .some(button => button.textContent.trim() === 'Fast scan'));
  const controls = await page.locator('button').evaluateAll(buttons => Object.fromEntries(
    ['Fast scan', 'Precision laser scan', 'Cancel scan and hold', 'Export captured scan', 'Clear captured scan']
      .map(label => { const button = buttons.find(item => item.textContent.trim() === label);
        return [label, button ? {disabled: button.disabled} : null]; })));
  for (const [label, value] of Object.entries(controls)) {
    if (!value) throw new Error('Missing scan control: ' + label);
  }
  if (controls['Cancel scan and hold'].disabled) throw new Error('Scan cancellation must remain available');
  const length = Number(await page.getByLabel('Downward sweep length (mm)', {exact: true}).inputValue());
  if (!Number.isFinite(length) || length < 1 || length > 100) {
    throw new Error('Downward sweep length must be bounded to 1–100 mm');
  }
  const text = await page.locator('body').innerText();
  if (!text.includes('1 mm intervals') || !text.includes('140 mm camera standoff')) {
    throw new Error('Fixed scan geometry explanation is missing');
  }
  return {result: 'PASS: scanning selector, fast/precision controls, fixed geometry, cancellation and persistent-cloud controls', controls};
}
