async page => {
  await page.setViewportSize({width: 1500, height: 1200});
  await page.reload();
  await page.getByRole('heading', {name: 'AutoWeld', exact: true}).waitFor();
  await page.getByText('OFFLINE SIMULATION', {exact: false}).waitFor();
  if (!await page.getByText('OFFLINE SIMULATION', {exact: false}).count()) {
    throw new Error('Browser smoke test refuses live hardware');
  }
  const enable = page.getByRole('button', {name: 'Enable motors', exact: true});
  const disable = page.getByRole('button', {name: 'Disable motors at rest', exact: true});
  if (await enable.isEnabled()) {
    await enable.click({force: true});
    await page.waitForFunction(() => {
      const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Enable motors');
      return b && b.disabled;
    }, null, {timeout: 10000});
  }
  await page.getByRole('button', {name: 'Save current pose', exact: true}).click({force: true});
  await page.getByRole('button', {name: 'Rest view', exact: true}).waitFor();
  const second = await page.context().newPage();
  try {
    await second.goto(page.url());
    await second.getByRole('button', {name: 'Rest view', exact: true}).waitFor();
    const stream = second.getByRole('checkbox', {name: 'Stream camera', exact: true});
    if (await stream.isChecked()) throw new Error('Offline camera unexpectedly enabled');
    await disable.click({force: true});
    await page.waitForFunction(() => {
      const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Enable motors');
      return b && !b.disabled;
    }, null, {timeout: 10000});
    await second.waitForFunction(() => {
      const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Enable motors');
      return b && !b.disabled;
    }, null, {timeout: 10000});
  } finally {
    await second.close();
  }
  return 'PASS: Viser UI, shared saved poses, two clients, opt-in camera and verified-rest disable';
}
