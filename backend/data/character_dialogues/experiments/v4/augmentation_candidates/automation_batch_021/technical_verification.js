const assert = require('node:assert/strict');

async function mapSequential(items, worker) {
  const results = [];
  for (const item of items) results.push(await worker(item));
  return results;
}

async function mapWithLimit(items, limit, worker) {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError('limit must be a positive integer');
  }
  const results = new Array(items.length);
  let nextIndex = 0;
  async function runner() {
    while (true) {
      const index = nextIndex++;
      if (index >= items.length) return;
      results[index] = await worker(items[index], index);
    }
  }
  const runnerCount = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: runnerCount }, () => runner()));
  return results;
}

async function test() {
  const visited = [];
  assert.deepEqual(await mapSequential([1, 2, 3], async n => {
    await new Promise(resolve => setTimeout(resolve, 2));
    visited.push(n);
    return n * 2;
  }), [2, 4, 6]);
  assert.deepEqual(visited, [1, 2, 3]);

  let active = 0;
  let maxActive = 0;
  const results = await mapWithLimit([30, 5, 20, 1], 2, async (ms, index) => {
    active += 1;
    maxActive = Math.max(maxActive, active);
    await new Promise(resolve => setTimeout(resolve, ms));
    active -= 1;
    return `task-${index}`;
  });
  assert.equal(maxActive, 2);
  assert.deepEqual(results, ['task-0', 'task-1', 'task-2', 'task-3']);
  assert.deepEqual(await mapWithLimit([], 2, async value => value), []);
  await assert.rejects(() => mapWithLimit([1], 0, async value => value), /positive integer/);
  console.log('technical verification passed');
}

test().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
