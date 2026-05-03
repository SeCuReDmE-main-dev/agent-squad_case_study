import { Triplet } from "./triplet";

export enum DecisionAction {
  CLARIFY = "CLARIFY",
  CONFIDENCE = "CONFIDENCE",
  CAVEAT = "CAVEAT",
  REJECT = "REJECT",
}

export interface DecisionThresholds {
  indeterminacy: number;
  falsity: number;
  truth: number;
}

export const DEFAULT_DECISION_THRESHOLDS: DecisionThresholds = {
  indeterminacy: 0.6,
  falsity: 0.5,
  truth: 0.7,
};

export function decide(
  value: Triplet,
  thresholds: DecisionThresholds = DEFAULT_DECISION_THRESHOLDS
): DecisionAction {
  new Triplet(thresholds.truth, thresholds.indeterminacy, thresholds.falsity);

  if (value.I > thresholds.indeterminacy) {
    return DecisionAction.CLARIFY;
  }
  if (value.F > thresholds.falsity) {
    return DecisionAction.REJECT;
  }
  if (value.T > thresholds.truth) {
    return DecisionAction.CONFIDENCE;
  }
  return DecisionAction.CAVEAT;
}
