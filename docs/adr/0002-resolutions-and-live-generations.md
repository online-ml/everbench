# Resolve observations and start live generations at promotion

Observations resolve to a known target or an explicitly unavailable target; both remain in weekly archives, while only known targets are scored and learned. Task-owned horizons and default outcomes replace negative-only deadlines and deletion of missing observations, preserving evidence of feed gaps.

Backtests and autonomous comparisons share one streaming replay built on River's `stream.simulate_qa`, the same scheduling primitive used by its progressive validation. River reveals due targets before the next observation, including equal timestamps. This replaces the backtest's separate timeline and retains only pending targets; unavailable targets are predicted but never scored or learned.

Promoted champions retain the state trained on their comparison archive and start accepting live observations at the promotion boundary. They do not replay the outgoing champion's live history or inherit its pending predictions. This intentionally trades catching up on the archive-to-promotion gap for bounded promotion cost, isolated generation metrics, and one sequence-based live checkpoint. Promotion and ingestion coordinate that boundary in the database.
