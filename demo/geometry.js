// Airfoil geometry and model inputs for the in-browser demo.
//
// A line-for-line port of airfrans.naca_generator, airfoil_uq.geometry, the feature functions
// in airfoil_uq.data and airfoil_uq.forces.integrate_coefficients, so the page feeds the model
// exactly what it saw in training. `nacaCase` mirrors `naca_case` in src/airfoil_uq/geometry.py.
// Points are [x, y] pairs; chord runs from x = 0 (leading edge) to x = 1 (trailing edge).
// Names follow the Python sources so the two can be read side by side.

/**
 * A 2D point or vector in chord units.
 * @typedef {[number, number]} Point
 */

/**
 * Surface geometry of one airfoil, in the layout `extract.process_case` stores for an AirfRANS case.
 * N is the number of surface points; the loop is closed, so there are also N segments.
 * @typedef {object} AirfoilGeometry
 * @property {number[]} naca - NACA parameters: [m, p, t] (4-digit) or [L, P, Q, t] (5-digit).
 * @property {Point[]} pos - (N) surface points, anticlockwise from the trailing edge over the upper surface.
 * @property {Point[]} normals - (N) inward unit normals per point, the average of the two adjacent segments.
 * @property {number[]} side - (N) +1 on the upper surface, -1 on the lower; leading and trailing edge are +1.
 * @property {number[]} s - (N) signed chord coordinate side * x, from -1 (lower TE) to +1 (upper TE).
 * @property {[number, number][]} lines - (N) segment endpoint indices; segment j runs from point j to j + 1.
 * @property {Point[]} cellNormals - (N) inward unit normal of each segment.
 * @property {number[]} cellLengths - (N) length of each segment.
 * @property {number[]} descriptor - (32) shape descriptor from `shapeDescriptor`.
 */

/**
 * Lift and drag coefficients with their pressure and viscous parts.
 * @typedef {object} ForceCoefficients
 * @property {number} cd - Total drag coefficient.
 * @property {number} cd_pressure - Drag from surface pressure.
 * @property {number} cd_viscous - Drag from skin friction.
 * @property {number} cl - Total lift coefficient.
 * @property {number} cl_pressure - Lift from surface pressure.
 * @property {number} cl_viscous - Lift from skin friction.
 */

/**
 * NACA thickness distribution with the closed trailing edge AirfRANS used.
 * @param {number} t - Maximum thickness as a fraction of chord.
 * @param {number} x - Chord position, 0 to 1.
 * @returns {number} Half-thickness at x.
 */
function thicknessDist(t, x) {
  return 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4);
}

/**
 * Camber line height and slope along the chord.
 * @param {number[]} params - 4-digit [m, p] (percent chord, tenths of chord) or 5-digit [L, P, Q]
 *   (design lift L with Cl = 0.15 L, camber position x = P / 20, Q = 1 for reflex camber).
 * @param {number[]} xs - Chord positions, 0 to 1.
 * @returns {{yc: Float64Array, dyc: Float64Array}} Camber height `yc` and slope `dyc` at each x.
 */
function camberLine(params, xs) {
  const yc = new Float64Array(xs.length), dyc = new Float64Array(xs.length);

  if (params.length === 2) {
    const m = params[0] / 100, p = params[1] / 10;
    if (p === 0 || p === 1) {
      // AirfRANS leaves the camber line flat here and only sets its slope
      xs.forEach((x, i) => { dyc[i] = p === 0 ? -2 * m * x : 2 * m * (1 - x); });
      return { yc, dyc };
    }
    xs.forEach((x, i) => {
      if (x < p) {
        yc[i] = (m / p ** 2) * (2 * p * x - x ** 2);
        dyc[i] = (2 * m / p ** 2) * (p - x);
      } else {
        yc[i] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x ** 2);
        dyc[i] = (2 * m / (1 - p) ** 2) * (p - x);
      }
    });
    return { yc, dyc };
  }

  const [l, p, q] = params;
  const cl = 3 / 20 * l, xf = p / 20;
  const f = (x) => x * (1 - Math.sqrt(x / 3)) - xf;
  const df = (x) => 1 - 3 * Math.sqrt(x / 3) / 2;
  let m = 0.5;
  for (;;) {
    const next = Math.max(m - f(m) / df(m), 0);
    const done = Math.abs(m - next) <= 1e-15;
    m = next;
    if (done) break;
  }
  const r = (3 * m - 7 * m ** 2 + 8 * m ** 3 - 4 * m ** 4) / Math.sqrt(m * (1 - m))
    - 3 / 2 * (1 - 2 * m) * (Math.PI / 2 - Math.asin(1 - 2 * m));
  const k1 = cl / r;

  if (q === 0) {
    xs.forEach((x, i) => {
      if (x <= m) {
        yc[i] = k1 * (x ** 3 - 3 * m * x ** 2 + m ** 2 * (3 - m) * x);
        dyc[i] = k1 * (3 * x ** 2 - 6 * m * x + m ** 2 * (3 - m));
      } else {
        yc[i] = k1 * m ** 3 * (1 - x);
        dyc[i] = -k1 * m ** 3;
      }
    });
  } else if (q === 1) {
    const k = (3 * (m - xf) ** 2 - m ** 3) / (1 - m) ** 3;
    xs.forEach((x, i) => {
      if (x <= m) {
        yc[i] = k1 * ((x - m) ** 3 - k * (1 - m) ** 3 * x - m ** 3 * x + m ** 3);
        dyc[i] = k1 * (3 * (x - m) ** 2 - k * (1 - m) ** 3 - m ** 3);
      } else {
        yc[i] = k1 * (k * (x - m) ** 3 - k * (1 - m) ** 3 * x - m ** 3 * x + m ** 3);
        dyc[i] = k1 * (3 * k * (x - m) ** 2 - k * (1 - m) ** 3 - m ** 3);
      }
    });
  } else {
    throw new Error("Q must be 0 for normal camber or 1 for reflex camber.");
  }

  return { yc, dyc };
}

/**
 * Closed surface loop from the trailing edge, over the upper surface, back along the lower one.
 * Unlike airfrans.naca_generator the trailing-edge point is not repeated at the end.
 * @param {number[]} params - NACA parameters: [m, p, t] (4-digit) or [L, P, Q, t] (5-digit), t in percent chord.
 * @param {number} nbSamples - Cosine-spaced samples per surface.
 * @returns {Point[]} 2 * nbSamples surface points.
 */
export function nacaGenerator(params, nbSamples) {
  const camber = params.length === 3 ? params.slice(0, 2) : params.slice(0, 3);
  const t = params[params.length - 1] / 100;
  const xs = Array.from({ length: nbSamples + 1 }, (_, i) => (1 - Math.cos(Math.PI * (1 - i / nbSamples))) / 2);
  const { yc, dyc } = camberLine(camber, xs);
  const upper = [], lower = [];
  xs.forEach((x, i) => {
    const yt = thicknessDist(t, x), theta = Math.atan(dyc[i]);
    upper.push([x - yt * Math.sin(theta), yc[i] + yt * Math.cos(theta)]);
    lower.push([x + yt * Math.sin(theta), yc[i] - yt * Math.cos(theta)]);
  });
  const pos = [...upper, ...lower.slice(0, -1).reverse()].slice(0, -1); // the last point repeated the trailing edge
  pos[0] = [1, 0];

  return pos;
}

/**
 * @param {number[]} values
 * @returns {number} Index of the first largest value.
 */
function argmax(values) {
  let best = 0;
  values.forEach((v, i) => { if (v > values[best]) best = i; });
  return best;
}

/**
 * @param {number[]} values
 * @returns {number} Index of the first smallest value.
 */
function argmin(values) {
  return argmax(values.map((v) => -v));
}

/**
 * @param {number[]} values
 * @returns {number} Arithmetic mean.
 */
function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/**
 * Linear interpolation like np.interp: clamped to the end values outside xp.
 * @param {number} x - Where to evaluate.
 * @param {number[]} xp - Sample positions, increasing.
 * @param {number[]} fp - Sample values.
 * @returns {number} Interpolated value at x.
 */
function interp(x, xp, fp) {
  if (x <= xp[0]) return fp[0];
  if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
  let lo = 0, hi = xp.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  return fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / (xp[hi] - xp[lo]);
}

/**
 * Upper and lower arcs as point indices, each from the trailing edge to the leading edge.
 * @param {Point[]} pos - Surface points.
 * @param {number[]} order - Point indices in walk order around the loop.
 * @returns {[number[], number[]]} [upper, lower]; the arc with the higher mean y is upper.
 */
function splitUpperLower(pos, order) {
  const te = argmax(order.map((i) => pos[i][0]));
  const rolled = [...order.slice(te), ...order.slice(0, te)];
  const le = argmin(rolled.map((i) => pos[i][0]));
  const arcA = rolled.slice(0, le + 1);
  const arcB = [...rolled.slice(le), rolled[0]].reverse();
  const meanY = (arc) => mean(arc.map((i) => pos[i][1]));

  return meanY(arcA) >= meanY(arcB) ? [arcA, arcB] : [arcB, arcA];
}

/**
 * Cosine-spaced chord positions, excluding the leading and trailing edge.
 * @param {number} nStations - Number of stations.
 * @returns {number[]} Chord positions in (0, 1), increasing.
 */
export function chordStations(nStations) {
  return Array.from({ length: nStations }, (_, i) => 0.5 * (1 - Math.cos(Math.PI * (i + 1) / (nStations + 1))));
}

/**
 * Shape descriptor for the branch input: surface heights at cosine-spaced chord stations.
 * @param {Point[]} pos - Surface points.
 * @param {number[]} upper - Upper-surface point indices.
 * @param {number[]} lower - Lower-surface point indices.
 * @param {number} [nStations=16] - Stations per surface.
 * @returns {number[]} Upper y values then lower y values, 2 * nStations in all.
 */
function shapeDescriptor(pos, upper, lower, nStations = 16) {
  const stations = chordStations(nStations);
  const out = [];
  for (const arc of [upper, lower]) {
    const byX = [...arc].sort((a, b) => pos[a][0] - pos[b][0]);
    const xp = byX.map((i) => pos[i][0]), fp = byX.map((i) => pos[i][1]);
    out.push(...stations.map((x) => interp(x, xp, fp)));
  }

  return out;
}

/**
 * Unit tangents (-ny, nx) from inward normals; they point towards increasing s.
 * @param {Point[]} normals - Inward normals.
 * @returns {Point[]} Unit tangents.
 */
export function tangent(normals) {
  return normals.map(([nx, ny]) => {
    const length = Math.hypot(nx, ny);
    return [-ny / length, nx / length];
  });
}

/**
 * Surface geometry of a NACA airfoil, in the layout of an extracted case (inward normals).
 * @param {number[]} naca - NACA parameters: [m, p, t] (4-digit) or [L, P, Q, t] (5-digit).
 * @param {number} [nbSamples=400] - Generator samples per surface; the loop has 2 * nbSamples points.
 * @returns {AirfoilGeometry}
 */
export function nacaCase(naca, nbSamples = 400) {
  const pos = nacaGenerator(naca, nbSamples);
  const n = pos.length;
  const lines = pos.map((_, i) => [i, (i + 1) % n]);
  const cellLengths = [], cellNormals = [];
  for (const [a, b] of lines) {
    const dx = pos[b][0] - pos[a][0], dy = pos[b][1] - pos[a][1], length = Math.hypot(dx, dy);
    cellLengths.push(length);
    cellNormals.push([-dy / length, dx / length]); // the loop runs anticlockwise, so this is inward
  }
  const normals = pos.map(() => [0, 0]);
  lines.forEach(([a, b], j) => {
    for (const i of [a, b]) {
      normals[i][0] += cellNormals[j][0];
      normals[i][1] += cellNormals[j][1];
    }
  });
  normals.forEach((normal) => {
    const length = Math.hypot(normal[0], normal[1]);
    normal[0] /= length;
    normal[1] /= length;
  });

  const order = pos.map((_, i) => i); // the walk order_loop finds on these lines
  const [upper, lower] = splitUpperLower(pos, order);
  const side = new Array(n).fill(0);
  lower.forEach((i) => { side[i] = -1; });
  upper.forEach((i) => { side[i] = 1; }); // LE / TE points shared by both arcs are labelled upper
  const s = pos.map(([x], i) => side[i] * x);

  return { naca, pos, normals, side, s, lines, cellNormals, cellLengths, descriptor: shapeDescriptor(pos, upper, lower) };
}

/**
 * Branch input for one flow condition. The exported model normalises these itself.
 * @param {number[]} descriptor - Shape descriptor from `nacaCase`.
 * @param {number} re - Reynolds number.
 * @param {number} aoaDeg - Angle of attack in degrees.
 * @returns {number[]} Shape descriptor, Re in millions, AoA in degrees.
 */
export function branchFeatures(descriptor, re, aoaDeg) {
  return [...descriptor, re / 1e6, aoaDeg];
}

/**
 * Per-point trunk inputs [u, x, 5*y, nx, ny, n.e, n x e], where u = sign(s) sqrt(|s|) and e is the
 * freestream direction. Written into `out` so a batch of AoAs can share one buffer.
 * @param {AirfoilGeometry} geometry - Airfoil surface.
 * @param {number} aoaDeg - Angle of attack in degrees.
 * @param {Float32Array} [out] - Destination buffer; a new one of length N * 7 by default.
 * @param {number} [offset=0] - Index in `out` to start writing at.
 * @returns {Float32Array} `out`.
 */
export function trunkFeatures(geometry, aoaDeg, out = new Float32Array(geometry.pos.length * 7), offset = 0) {
  const { pos, s } = geometry;
  const normals = geometry.normals.slice();
  const bySDescending = s.map((_, i) => i).sort((a, b) => s[b] - s[a]);
  normals[bySDescending[0]] = normals[bySDescending[1]]; // the trailing edge takes its neighbour's normal
  const aoa = aoaDeg * Math.PI / 180, ex = Math.cos(aoa), ey = Math.sin(aoa);
  pos.forEach(([x, y], i) => {
    const [nx, ny] = normals[i], o = offset + 7 * i;
    out[o] = Math.sign(s[i]) * Math.sqrt(Math.abs(s[i]));
    out[o + 1] = x;
    out[o + 2] = 5 * y;
    out[o + 3] = nx;
    out[o + 4] = ny;
    out[o + 5] = nx * ex + ny * ey;
    out[o + 6] = nx * ey - ny * ex;
  });

  return out;
}

/**
 * Signed skin friction, positive for flow from the leading edge towards the trailing edge.
 * @param {Float64Array} cfT - Skin friction along the tangent, which points towards increasing s.
 * @param {number[]} side - +1 upper / -1 lower per point, from `nacaCase`.
 * @returns {Float64Array} Streamwise skin friction per point.
 */
export function streamwiseCf(cfT, side) {
  return cfT.map((value, i) => (side[i] >= 0 ? 1 : -1) * value);
}

/**
 * Lift and drag from per-point Cp and tangential Cf, integrated segment by segment.
 * @param {AirfoilGeometry} geometry - Airfoil surface.
 * @param {ArrayLike<number>} cp - Pressure coefficient per point.
 * @param {ArrayLike<number>} cfT - Tangential skin friction per point (see `streamwiseCf`).
 * @param {number} aoaDeg - Angle of attack in degrees.
 * @returns {ForceCoefficients}
 */
export function integrateCoefficients(geometry, cp, cfT, aoaDeg) {
  const tangents = tangent(geometry.normals);
  const cf = tangents.map(([tx, ty], i) => [cfT[i] * tx, cfT[i] * ty]);
  let px = 0, py = 0, vx = 0, vy = 0;
  geometry.lines.forEach(([a, b], j) => {
    const length = geometry.cellLengths[j], [nx, ny] = geometry.cellNormals[j];
    const cpCell = 0.5 * (cp[a] + cp[b]);
    px += cpCell * nx * length;
    py += cpCell * ny * length;
    vx += 0.5 * (cf[a][0] + cf[b][0]) * length;
    vy += 0.5 * (cf[a][1] + cf[b][1]) * length;
  });
  const aoa = aoaDeg * Math.PI / 180, c = Math.cos(aoa), sn = Math.sin(aoa);
  const cdPressure = c * px + sn * py, clPressure = -sn * px + c * py;
  const cdViscous = c * vx + sn * vy, clViscous = -sn * vx + c * vy;

  return {
    cd: cdPressure + cdViscous, cd_pressure: cdPressure, cd_viscous: cdViscous,
    cl: clPressure + clViscous, cl_pressure: clPressure, cl_viscous: clViscous,
  };
}
