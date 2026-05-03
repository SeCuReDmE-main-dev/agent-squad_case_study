import { nConorm } from "./operators";
import { Triplet } from "./triplet";

export function neutrosophicConsensus(responses: Iterable<Triplet>): Triplet {
  const iterator = responses[Symbol.iterator]();
  const first = iterator.next();
  if (first.done) {
    throw new Error("responses must contain at least one triplet");
  }

  let consensus = first.value;
  for (let next = iterator.next(); !next.done; next = iterator.next()) {
    consensus = nConorm(consensus, next.value);
  }

  return consensus;
}

export function neutrosophicEvidenceConsensus(responses: Iterable<Triplet>): Triplet {
  const triplets = Array.from(responses);
  if (!triplets.length) {
    throw new Error("responses must contain at least one triplet");
  }

  const formalUnion = neutrosophicConsensus(triplets);
  const truthValues = triplets.map((response) => response.T);
  const falsityValues = triplets.map((response) => response.F);
  const truthSpread = Math.max(...truthValues) - Math.min(...truthValues);
  const falsitySpread = Math.max(...falsityValues) - Math.min(...falsityValues);
  const contradiction = Math.min(Math.max(...truthValues), Math.max(...falsityValues));
  const conflict = Math.max(truthSpread, falsitySpread, contradiction);

  return new Triplet(
    formalUnion.T,
    Math.max(formalUnion.I, conflict),
    Math.max(...falsityValues)
  );
}
