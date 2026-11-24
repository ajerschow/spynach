// Cells of a playground example, read from and written back to its Python source.
//
// A line that starts with "# %%" begins a cell, and the text after the marker is the title of the cell:
//
//     # %% Free evolution under shifts and a coupling
//     rho = I(1, "x").evolve(H, t)
//     rho
//
// What stands before the first marker is a cell without a title. The source stays the only state: the page
// builds the cells from it and writes an edited cell back, so the same text runs as one script (in the plain
// editor and in the tests) and cell by cell. No dependencies, so the same file runs in the page and under Node
// (tests/unit/test_playground_cells.py).

const MARKER = /^# %%(.*)$/;

/** Whether a source is written in cells: it has a marker line. */
export function hasCells(source) {
  return source.split("\n").some((line) => MARKER.test(line));
}

/** The cells of a source in order: {title, code}. Blank lines at the two ends of a cell are dropped, and so is
 *  an empty stretch before the first marker. */
export function splitCells(source) {
  const cells = [];
  let current = { title: "", lines: [], marked: false };
  for (const line of source.replace(/\r\n?/g, "\n").split("\n")) {
    const match = MARKER.exec(line);
    if (match) {
      cells.push(current);
      current = { title: match[1].trim(), lines: [], marked: true };
    } else {
      current.lines.push(line);
    }
  }
  cells.push(current);
  return cells
    .map((cell) => ({ title: cell.title, code: cell.lines.join("\n").replace(/^\n+/, "").replace(/\s+$/, ""), marked: cell.marked }))
    .filter((cell) => cell.marked || cell.code !== "")
    .map(({ title, code }) => ({ title, code }));
}

/** The source of a list of cells: every cell under its marker line, one blank line between cells. */
export function joinCells(cells) {
  return cells
    .map((cell) => ("# %%" + (cell.title ? " " + cell.title : "") + "\n" + cell.code).replace(/\s+$/, ""))
    .join("\n\n");
}
