import { Triplet } from "./triplet";

export function nNorm(left: Triplet, right: Triplet): Triplet {
  return new Triplet(
    Math.min(left.T, right.T),
    Math.max(left.I, right.I),
    Math.max(left.F, right.F)
  );
}

export function nConorm(left: Triplet, right: Triplet): Triplet {
  return new Triplet(
    Math.max(left.T, right.T),
    Math.min(left.I, right.I),
    Math.min(left.F, right.F)
  );
}

export function negate(value: Triplet): Triplet {
  return new Triplet(value.F, value.I, value.T);
}
