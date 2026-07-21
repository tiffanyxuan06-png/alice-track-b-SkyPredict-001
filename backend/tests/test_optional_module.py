"""Optional advanced track: risk-based maintenance prioritization."""

from __future__ import annotations


def test_prioritize_ranks_by_urgency(client, batch_readings):
    r = client.post("/prioritize", json={"readings": batch_readings})
    assert r.status_code == 200
    body = r.json()
    ranked = body["ranked_engines"]

    assert len(ranked) == len(batch_readings)
    # ranks are 1..N, contiguous
    assert [e["priority_rank"] for e in ranked] == list(range(1, len(ranked) + 1))
    # most urgent first => predicted RUL ascending
    ruls = [e["predicted_rul"] for e in ranked]
    assert ruls == sorted(ruls)
    assert body["summary"]["total_engines"] == len(batch_readings)
