"""StrategyConfig — the single source of truth for every RSI(2) rule (spec §8)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields


@dataclass(frozen=True)
class StrategyConfig:
    trend_len: int = 200
    trend_type: str = "SMA"          # "SMA" | "EMA"
    rsi_len: int = 2
    entry_level: float = 10.0
    entry_delay: int = 1             # bars from signal to fill
    recheck_filter: bool = False     # cancel entry if close < trend on the entry bar
    fast_len: int = 5
    exit_level: float = 65.0
    exit_logic: str = "AND"          # "AND" | "OR"
    use_trend_stop: bool = True      # exit on a close below the trend MA
    use_stop: bool = False           # intrabar % stop
    stop_pct: float = 8.0
    use_timeout: bool = False
    max_bars: int = 10
    flag_level: float = 10.0         # research tag only (§6)
    commission_bps: float = 3.0      # per side
    slippage_bps: float = 1.0        # per side
    initial_capital: float = 100_000.0
    position_pct: float = 100.0      # % of equity per trade
    whole_shares: bool = False
    start_date: str | None = None    # inclusive; gates entries, starts the equity curve
    end_date: str | None = None      # inclusive; data after it is ignored

    def __post_init__(self) -> None:
        for name in ("trend_len", "rsi_len", "fast_len", "max_bars"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer (got {v!r})")
        for name in ("entry_level", "exit_level", "flag_level"):
            v = getattr(self, name)
            if not 0 <= v <= 100:
                raise ValueError(f"{name} must be between 0 and 100 (got {v!r})")
        if not isinstance(self.entry_delay, int) or isinstance(self.entry_delay, bool) or self.entry_delay < 0:
            raise ValueError(f"entry_delay must be an integer >= 0 (got {self.entry_delay!r})")
        if self.position_pct <= 0:
            raise ValueError(f"position_pct must be > 0 (got {self.position_pct!r})")
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0 (got {self.initial_capital!r})")
        if not 0 < self.stop_pct < 100:
            raise ValueError(f"stop_pct must be between 0 and 100 (got {self.stop_pct!r})")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be >= 0")
        if self.exit_logic not in ("AND", "OR"):
            raise ValueError(f"exit_logic must be 'AND' or 'OR' (got {self.exit_logic!r})")
        if self.trend_type not in ("SMA", "EMA"):
            raise ValueError(f"trend_type must be 'SMA' or 'EMA' (got {self.trend_type!r})")

    @property
    def total_bps(self) -> float:
        """Cost per side in basis points (commission + slippage)."""
        return self.commission_bps + self.slippage_bps

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]
