// Grid indices refer to the shared endpoint-aligned display lattice.
export function checkedSpacing(value) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1 || n > 65536)
    throw new Error(
      "Grid spacing must be an integer from 1 to 65536 display voxels",
    );
  return n;
}
export function rasterValue(x, z, nx, nz, spacing) {
  x = Math.max(0, Math.min(nx - 1, x));
  z = Math.max(0, Math.min(nz - 1, z));
  const x0 = Math.floor(x),
    z0 = Math.floor(z),
    fx = x - x0,
    fz = z - z0;
  const g = (i, j) => (i % spacing === 0 || j % spacing === 0 ? 1 : 0);
  return (
    (1 - fz) * ((1 - fx) * g(x0, z0) + fx * g(Math.min(x0 + 1, nx - 1), z0)) +
    fz *
      ((1 - fx) * g(x0, Math.min(z0 + 1, nz - 1)) +
        fx * g(Math.min(x0 + 1, nx - 1), Math.min(z0 + 1, nz - 1)))
  );
}
export function meshSegments(nx, nz, spacing) {
  const pairs = [];
  for (let x = 0; x < nx; x += spacing)
    for (let z = 0; z < nz - 1; z++) pairs.push(x, z, x, z + 1);
  for (let z = 0; z < nz; z += spacing)
    for (let x = 0; x < nx - 1; x++) pairs.push(x, z, x + 1, z);
  return new Float32Array(pairs);
}
