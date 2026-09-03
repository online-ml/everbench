let updatedAt = Date.now();
const updated = document.querySelector('[data-updated]');
const sortStates = new Map();
let modelDetailRequest;

function updateAge() {
  const seconds = Math.max(0, (Date.now() - updatedAt) / 1000);
  updated.textContent = `updated ${seconds.toFixed(1)}s ago`;
}

function updateRelativeTimes() {
  document.querySelectorAll('[data-relative-time]').forEach(time => {
    const seconds = Math.max(0, (Date.now() - Date.parse(time.dateTime)) / 1000);
    const units = seconds < 60
      ? [seconds, 's']
      : seconds < 3_600
        ? [seconds / 60, 'm']
        : seconds < 86_400
          ? [seconds / 3_600, 'h']
          : [seconds / 86_400, 'd'];
    time.textContent = `${Math.floor(units[0])}${units[1]} ago`;
    time.title = new Date(time.dateTime).toLocaleString();
  });
}

function sortTable(table) {
  const state = sortStates.get(table.dataset.sortTable);
  if (!state) return;

  const rows = [...table.tBodies[0].rows];
  const compare = (left, right) => {
    const leftValue = left.cells[state.column].dataset.sortValue ?? left.cells[state.column].textContent.trim();
    const rightValue = right.cells[state.column].dataset.sortValue ?? right.cells[state.column].textContent.trim();
    if (state.type === 'text') return leftValue.localeCompare(rightValue) * state.direction;
    const leftNumber = state.type === 'date' ? Date.parse(leftValue) : Number(leftValue);
    const rightNumber = state.type === 'date' ? Date.parse(rightValue) : Number(rightValue);
    if (!Number.isFinite(leftNumber) && !Number.isFinite(rightNumber)) return 0;
    if (!Number.isFinite(leftNumber)) return 1;
    if (!Number.isFinite(rightNumber)) return -1;
    return (leftNumber - rightNumber) * state.direction;
  };
  rows.sort(compare).forEach(row => table.tBodies[0].append(row));

  table.querySelectorAll('[data-sort-direction]').forEach(button => button.removeAttribute('data-sort-direction'));
  table.tHead.rows[0].cells[state.column].querySelector('[data-sort-type]').dataset.sortDirection =
    state.direction === 1 ? 'ascending' : 'descending';
}

function restoreSorting(root = document) {
  root.querySelectorAll('[data-sortable]').forEach(sortTable);
}

document.body.addEventListener('htmx:afterSwap', event => {
  if (event.detail.target.id === 'task-panel') {
    updatedAt = Date.now();
    updateAge();
  }
  updateRelativeTimes();
  restoreSorting();
});

document.addEventListener('click', async event => {
  const sortControl = event.target.closest('[data-sort-type]');
  if (sortControl) {
    const table = sortControl.closest('[data-sortable]');
    const column = sortControl.closest('th').cellIndex;
    const current = sortStates.get(table.dataset.sortTable);
    sortStates.set(table.dataset.sortTable, {
      column,
      type: sortControl.dataset.sortType,
      direction: current?.column === column ? -current.direction : 1,
    });
    sortTable(table);
    return;
  }

  const trigger = event.target.closest('[data-model-dialog], [data-model-dialog-url]');
  if (!trigger) return;

  const dialog = document.querySelector('[data-persistent-model-dialog]');
  dialog.querySelector('[data-model-dialog-heading]').textContent = trigger.dataset.modelDialogHeading;
  const content = dialog.querySelector('[data-model-dialog-content]');
  if (trigger.dataset.modelDialogUrl) {
    modelDetailRequest?.abort();
    modelDetailRequest = new AbortController();
    const request = modelDetailRequest;
    content.textContent = 'Loading…';
    if (!dialog.open) dialog.showModal();
    try {
      const response = await fetch(trigger.dataset.modelDialogUrl, { signal: request.signal });
      const markup = response.ok
        ? await response.text()
        : '<p class="muted">Model details are unavailable.</p>';
      if (modelDetailRequest !== request) return;
      content.innerHTML = markup;
    } catch (error) {
      if (error.name === 'AbortError' || modelDetailRequest !== request) return;
      content.innerHTML = '<p class="muted">Model details are unavailable.</p>';
    }
  } else {
    modelDetailRequest?.abort();
    const template = trigger.nextElementSibling;
    if (!(template instanceof HTMLTemplateElement)) return;
    content.replaceChildren(template.content.cloneNode(true));
  }
  updateRelativeTimes();
  if (!dialog.open) dialog.showModal();
});

updateRelativeTimes();
setInterval(() => {
  updateAge();
  updateRelativeTimes();
}, 1000);
