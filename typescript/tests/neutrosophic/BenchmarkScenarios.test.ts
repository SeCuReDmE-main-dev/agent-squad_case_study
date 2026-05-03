import {
  DecisionAction,
  decide,
  neutrosophicConsensus,
  neutrosophicEvidenceConsensus,
  scoreTextResponse,
  Triplet,
} from "../../src/neutrosophic";

describe("neutrosophic benchmark scenarios", () => {
  test("conflict preservation separates formal union from evidence fusion", () => {
    const clearSupport = new Triplet(0.92, 0.08, 0.02);
    const strongCounterEvidence = new Triplet(0.18, 0.18, 0.84);
    const weakSupport = new Triplet(0.64, 0.22, 0.08);

    const formalUnion = neutrosophicConsensus([
      clearSupport,
      strongCounterEvidence,
      weakSupport,
    ]);
    const evidenceFusion = neutrosophicEvidenceConsensus([
      clearSupport,
      strongCounterEvidence,
      weakSupport,
    ]);

    expect(formalUnion).toEqual(new Triplet(0.92, 0.08, 0.02));
    expect(evidenceFusion.T).toBe(0.92);
    expect(evidenceFusion.F).toBe(0.84);
    expect(evidenceFusion.I).toBeGreaterThanOrEqual(0.84);
    expect(decide(evidenceFusion)).toBe(DecisionAction.CLARIFY);
  });

  test("scorer calibration keeps expected response bands", () => {
    const direct = scoreTextResponse(
      "The billing agent should answer because the user asks about invoices, refunds, and account balance details."
    );
    const hedged = scoreTextResponse(
      "Maybe it could be billing, but it depends because the request is unclear and possibly ambiguous."
    );
    const refusal = scoreTextResponse("Sorry, but I cannot comply with this request.");
    const error = scoreTextResponse("The task failed with an invalid response and an error.");

    expect(direct.T).toBeGreaterThan(hedged.T);
    expect(hedged.I).toBeGreaterThan(direct.I);
    expect(refusal.F).toBeGreaterThanOrEqual(0.5);
    expect(error.F).toBeGreaterThan(direct.F);
    expect(decide(hedged)).toBe(DecisionAction.CLARIFY);
  });
});
