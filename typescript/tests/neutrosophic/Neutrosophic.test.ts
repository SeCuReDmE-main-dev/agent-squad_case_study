import {
  decide,
  DecisionAction,
  nConorm,
  nNorm,
  negate,
  neutrosophicConsensus,
  neutrosophicEvidenceConsensus,
  scoreClassifierConfidence,
  scoreTextResponse,
  Triplet,
} from "../../src/neutrosophic";

describe("neutrosophic core", () => {
  test("validates triplet scores", () => {
    expect(new Triplet(0.2, 0.7, 0.1)).toEqual({ T: 0.2, I: 0.7, F: 0.1 });
    expect(() => new Triplet(1.1, 0, 0)).toThrow("between 0 and 1");
  });

  test("applies formal operators", () => {
    const left = new Triplet(0.8, 0.4, 0.1);
    const right = new Triplet(0.3, 0.7, 0.6);

    expect(nNorm(left, right)).toEqual(new Triplet(0.3, 0.7, 0.6));
    expect(nConorm(left, right)).toEqual(new Triplet(0.8, 0.4, 0.1));
    expect(negate(left)).toEqual(new Triplet(0.1, 0.4, 0.8));
  });

  test("aggregates formal and evidence consensus", () => {
    const clear = new Triplet(0.9, 0.1, 0.0);
    const conflicting = new Triplet(0.2, 0.2, 0.8);

    expect(neutrosophicConsensus([clear, conflicting])).toEqual(new Triplet(0.9, 0.1, 0.0));

    const evidence = neutrosophicEvidenceConsensus([clear, conflicting]);
    expect(evidence.T).toBe(0.9);
    expect(evidence.I).toBeGreaterThan(0.6);
    expect(evidence.F).toBe(0.8);
  });

  test("decides by priority thresholds", () => {
    expect(decide(new Triplet(0.2, 0.8, 0.9))).toBe(DecisionAction.CLARIFY);
    expect(decide(new Triplet(0.2, 0.1, 0.9))).toBe(DecisionAction.REJECT);
    expect(decide(new Triplet(0.9, 0.1, 0.0))).toBe(DecisionAction.CONFIDENCE);
    expect(decide(new Triplet(0.4, 0.2, 0.0))).toBe(DecisionAction.CAVEAT);
  });

  test("scores text and classifier confidence safely", () => {
    const direct = scoreTextResponse("The answer is clear and complete for billing invoices.");
    const hedged = scoreTextResponse("Maybe it could be billing, but it depends and is unclear.");
    const failed = scoreTextResponse("The task failed with an invalid response and an error.");

    expect(hedged.I).toBeGreaterThan(direct.I);
    expect(failed.F).toBeGreaterThan(direct.F);
    expect(scoreClassifierConfidence(0.8, true)).toEqual(new Triplet(0.8, 0.19999999999999996, 0));
    expect(() => scoreClassifierConfidence(Number.NaN, true)).toThrow("confidence must be finite");
  });
});
