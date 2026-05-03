from agent_squad.neutrosophic import (
    DecisionAction,
    Triplet,
    decide,
    neutrosophic_consensus,
    neutrosophic_evidence_consensus,
    score_text_response,
)


def test_conflict_preservation_benchmark_separates_formal_union_from_evidence_fusion():
    clear_support = Triplet(T=0.92, I=0.08, F=0.02)
    strong_counter_evidence = Triplet(T=0.18, I=0.18, F=0.84)
    weak_support = Triplet(T=0.64, I=0.22, F=0.08)

    formal_union = neutrosophic_consensus([
        clear_support,
        strong_counter_evidence,
        weak_support,
    ])
    evidence_fusion = neutrosophic_evidence_consensus([
        clear_support,
        strong_counter_evidence,
        weak_support,
    ])

    assert formal_union == Triplet(T=0.92, I=0.08, F=0.02)
    assert evidence_fusion.T == 0.92
    assert evidence_fusion.F == 0.84
    assert evidence_fusion.I >= 0.84
    assert decide(evidence_fusion) == DecisionAction.CLARIFY


def test_scorer_calibration_benchmark_keeps_expected_response_bands():
    direct = score_text_response(
        "The billing agent should answer because the user asks about invoices, refunds, and account balance details."
    )
    hedged = score_text_response(
        "Maybe it could be billing, but it depends because the request is unclear and possibly ambiguous."
    )
    refusal = score_text_response("Sorry, but I cannot comply with this request.")
    error = score_text_response("The task failed with an invalid response and an error.")

    assert direct.T > hedged.T
    assert hedged.I > direct.I
    assert refusal.F >= 0.5
    assert error.F > direct.F
    assert decide(hedged) == DecisionAction.CLARIFY
