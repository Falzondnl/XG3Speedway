"""
XG3 Speedway GP — Feature Engineering.

Schema notes (from data analysis):
- round_result_rankings: one row per rider per round, 'rank' = final GP position (1=winner)
- heat_rankings: one row per rider per heat, 'points' = 3/2/1/0, 'title' = heat name
- round_details: venue info (country, track_length, city)
- riders: career stats (sgpWins, heatsWon, heatsRaced, appearances, wins, fimRank, bornAt)

Features per (rider, round) observation — all computed BEFORE the round using
historical data strictly prior to that round's date.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ELO Engine (Speedway-specific)
# ---------------------------------------------------------------------------

class SpeedwayELO:
    """Simple head-to-head ELO adapted for multi-rider heat format.

    For each heat: 1st place beats 2nd, 3rd, 4th.  2nd beats 3rd, 4th.
    3rd beats 4th.  All pairwise comparisons within one heat.
    K-factor fixed at 32.
    """

    def __init__(self, k: float = 32.0, initial: float = 1500.0) -> None:
        self.k = k
        self.initial = initial
        self.ratings: dict[str, float] = {}

    def get(self, rider_slug: str) -> float:
        return self.ratings.get(rider_slug, self.initial)

    def update_heat(self, results: list[tuple[str, int]]) -> None:
        """results: list of (rider_slug, rank_in_heat) sorted by rank asc."""
        if len(results) < 2:
            return
        riders = [r[0] for r in results]
        ranks = [r[1] for r in results]

        # Store snapshots before updating
        pre = {slug: self.get(slug) for slug in riders}

        delta: dict[str, float] = {s: 0.0 for s in riders}
        for i in range(len(riders)):
            for j in range(i + 1, len(riders)):
                ri, rj = riders[i], riders[j]
                # rider i placed better than rider j (lower rank = better)
                if ranks[i] < ranks[j]:
                    expected_i = 1.0 / (1.0 + 10 ** ((pre[rj] - pre[ri]) / 400.0))
                    delta[ri] += self.k * (1.0 - expected_i)
                    delta[rj] += self.k * (0.0 - (1.0 - expected_i))
                elif ranks[j] < ranks[i]:
                    expected_j = 1.0 / (1.0 + 10 ** ((pre[ri] - pre[rj]) / 400.0))
                    delta[rj] += self.k * (1.0 - expected_j)
                    delta[ri] += self.k * (0.0 - (1.0 - expected_j))

        for slug in riders:
            self.ratings[slug] = pre[slug] + delta[slug]

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class SpeedwayFeatureExtractor:
    """Build per-(rider, round) feature DataFrame from raw CSVs.

    All features for a given round are computed using data STRICTLY before
    that round's start date (temporal ordering enforced).
    """

    ROUND_POINTS_MAP = {1: 20, 2: 18, 3: 16, 4: 14, 5: 12, 6: 11,
                        7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 5,
                        13: 4, 14: 3, 15: 2, 16: 1}

    def __init__(self) -> None:
        self._fitted = False

    def fit_transform(
        self,
        round_results: pd.DataFrame,
        heat_rankings: pd.DataFrame,
        round_details: pd.DataFrame,
        riders: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build full feature dataset with target column.

        Returns DataFrame with one row per (rider, round) with:
          - features (all pre-round)
          - target: is_winner (1 if rank==1 in round_results)
          - round_id, rider_slug, round_starts_at (for temporal splitting)
        """
        logger.info("Building Speedway feature dataset...")

        # ---- Prepare round_results ----------------------------------------
        rr = self._clean_round_results(round_results)
        # ---- Prepare heat_rankings ----------------------------------------
        hr = self._clean_heat_rankings(heat_rankings)
        # ---- Prepare round_details ----------------------------------------
        rd = self._clean_round_details(round_details)
        # ---- Prepare static rider stats -----------------------------------
        rider_static = self._clean_riders(riders)

        # ---- Build ELO ratings chronologically ----------------------------
        elo_snapshots, final_elo = self._build_elo_snapshots_with_final(hr, rr)

        # ---- Merge everything and build rolling features ------------------
        rows = []
        rounds_sorted = rr.sort_values("round_starts_at").round_id.unique()

        # Accumulate rolling stats per rider
        rider_history: dict[str, dict] = {}  # slug -> {heat_pts: [], round_ranks: [], venues: {}}

        for round_id in rounds_sorted:
            rr_round = rr[rr.round_id == round_id]
            if rr_round.empty:
                continue

            round_date = rr_round["round_starts_at"].iloc[0]
            rd_round = rd[rd.round_id == round_id].iloc[0] if not rd[rd.round_id == round_id].empty else None
            venue_slug = rd_round["venue_slug"] if rd_round is not None else None
            venue_country = rd_round["venue_country"] if rd_round is not None else None
            track_length = rd_round["track_length"] if rd_round is not None else np.nan
            season_id = rr_round["season_id"].iloc[0]

            # ELO snapshot BEFORE this round
            elo_snap = elo_snapshots.get(round_id, {})

            # All heats in this round (for in-round heat features — not used for target)
            hr_round = hr[hr.round_id == round_id]

            # Build a row per rider in this round
            for _, rider_row in rr_round.iterrows():
                slug = rider_row["rider_slug"]
                rank = rider_row["rank"]
                is_winner = 1 if rank == 1 else 0

                hist = rider_history.get(slug, {
                    "heat_pts": [],
                    "round_ranks": [],
                    "round_wins": 0,
                    "venues": {},
                    "season_round_pts": {},
                    "n_rounds": 0,
                })

                elo_rating = elo_snap.get(slug, 1500.0)

                # Static career stats (snapshot from riders CSV for that season)
                static = rider_static.get(slug, {})

                # Rolling pre-round features
                feat = self._compute_features(
                    slug=slug,
                    hist=hist,
                    elo_rating=elo_rating,
                    venue_slug=venue_slug,
                    venue_country=venue_country,
                    track_length=track_length,
                    rider_row=rider_row,
                    static=static,
                    season_id=season_id,
                )
                feat["round_id"] = round_id
                feat["season_id"] = season_id
                feat["rider_slug"] = slug
                feat["round_starts_at"] = round_date
                feat["is_winner"] = is_winner
                feat["final_rank"] = rank
                rows.append(feat)

            # ---- Update history AFTER building features (no leakage) ------
            for _, rider_row in rr_round.iterrows():
                slug = rider_row["rider_slug"]
                rank = rider_row["rank"]
                hist = rider_history.get(slug, {
                    "heat_pts": [],
                    "round_ranks": [],
                    "round_wins": 0,
                    "venues": {},
                    "season_round_pts": {},
                    "n_rounds": 0,
                })
                # Aggregate heat points in this round for this rider
                rider_heats = hr_round[hr_round["rider_slug"] == slug]
                heat_pt_total = rider_heats["points"].sum() if not rider_heats.empty else 0
                hist["heat_pts"].append(heat_pt_total)
                hist["round_ranks"].append(rank)
                if rank == 1:
                    hist["round_wins"] += 1
                hist["n_rounds"] += 1
                if venue_slug:
                    hist["venues"][venue_slug] = hist["venues"].get(venue_slug, [])
                    hist["venues"][venue_slug].append(rank)
                rider_history[slug] = hist

        df = pd.DataFrame(rows)
        logger.info("Feature dataset built: %d rows, %d rounds, %d riders",
                    len(df), df["round_id"].nunique(), df["rider_slug"].nunique())
        self._fitted = True
        # Persist final ELO + rider history for inference after training
        self._elo = final_elo
        self._rider_history = rider_history
        return df

    # -----------------------------------------------------------------------
    def _compute_features(
        self,
        slug: str,
        hist: dict,
        elo_rating: float,
        venue_slug: Optional[str],
        venue_country: Optional[str],
        track_length: float,
        rider_row: pd.Series,
        static: dict,
        season_id: int,
    ) -> dict:
        n = hist["n_rounds"]
        round_ranks = hist["round_ranks"]
        heat_pts = hist["heat_pts"]

        # --- ELO ---
        feat: dict = {
            "elo_rating": elo_rating,
            "elo_minus_1500": elo_rating - 1500.0,
        }

        # --- Recent form (last 5 rounds) ---
        last5_ranks = round_ranks[-5:] if n >= 1 else []
        last5_heat = heat_pts[-5:] if n >= 1 else []
        feat["n_rounds_competed"] = n
        feat["avg_rank_last5"] = float(np.mean(last5_ranks)) if last5_ranks else np.nan
        feat["avg_heat_pts_last5"] = float(np.mean(last5_heat)) if last5_heat else np.nan
        feat["min_rank_last5"] = float(np.min(last5_ranks)) if last5_ranks else np.nan
        feat["win_rate_last5"] = sum(1 for r in last5_ranks if r == 1) / len(last5_ranks) if last5_ranks else np.nan
        feat["top3_rate_last5"] = sum(1 for r in last5_ranks if r <= 3) / len(last5_ranks) if last5_ranks else np.nan
        feat["last_round_rank"] = float(round_ranks[-1]) if round_ranks else np.nan
        feat["last_heat_pts"] = float(heat_pts[-1]) if heat_pts else np.nan

        # --- Career form ---
        feat["career_win_rate"] = hist["round_wins"] / n if n > 0 else np.nan
        feat["career_avg_rank"] = float(np.mean(round_ranks)) if round_ranks else np.nan

        # --- Venue / track familiarity ---
        venue_history = hist["venues"].get(venue_slug, []) if venue_slug else []
        feat["venue_appearances"] = len(venue_history)
        feat["venue_avg_rank"] = float(np.mean(venue_history)) if venue_history else np.nan
        feat["venue_win"] = int(any(r == 1 for r in venue_history))
        feat["track_length"] = track_length if not pd.isna(track_length) else np.nan

        # --- Country advantage ---
        rider_country = rider_row.get("rider_country", "")
        feat["home_country"] = int(str(rider_country) == str(venue_country)) if venue_country else 0

        # --- Wildcard / substitute ---
        feat["is_wildcard"] = int(rider_row.get("is_wildcard", False))
        feat["is_substitute"] = int(rider_row.get("is_substitute", False))

        # --- Career static stats (from riders CSV) ---
        feat["career_sgp_wins"] = static.get("sgpWins", np.nan)
        feat["career_appearances"] = static.get("appearances", np.nan)
        feat["career_heats_won"] = static.get("heatsWon", np.nan)
        feat["career_heats_raced"] = static.get("heatsRaced", np.nan)
        feat["career_finals"] = static.get("finals", np.nan)
        feat["career_wins"] = static.get("wins", np.nan)
        feat["fim_rank"] = static.get("fimRank", np.nan)

        # Derived career metrics
        if static.get("heatsRaced") and static.get("heatsRaced", 0) > 0:
            feat["career_heat_win_rate"] = (
                (static.get("heatsWon") or 0) / static["heatsRaced"]
            )
        else:
            feat["career_heat_win_rate"] = np.nan

        # --- Rider age at round date ---
        born_at = static.get("bornAt")
        feat["rider_age"] = np.nan
        if born_at and not pd.isna(born_at):
            try:
                round_date = rider_row.get("round_starts_at")
                if round_date and not pd.isna(round_date):
                    born_dt = pd.to_datetime(born_at, errors="coerce")
                    round_dt = pd.to_datetime(round_date, errors="coerce")
                    if born_dt is not pd.NaT and round_dt is not pd.NaT:
                        feat["rider_age"] = (round_dt - born_dt).days / 365.25
            except Exception:
                pass

        # --- Season context ---
        feat["season_id"] = season_id

        return feat

    # -----------------------------------------------------------------------
    def _build_elo_snapshots_with_final(
        self, hr: pd.DataFrame, rr: pd.DataFrame
    ) -> tuple[dict[int, dict[str, float]], "SpeedwayELO"]:
        """
        Walk rounds chronologically; update ELO from heat results.
        Returns ({round_id: elo_snapshot_before_round}, final_elo_instance).
        The final ELO instance is saved for inference use after training.
        """
        elo = SpeedwayELO()
        snapshots: dict[int, dict[str, float]] = {}

        rounds_sorted = (
            rr[["round_id", "round_starts_at"]]
            .drop_duplicates()
            .sort_values("round_starts_at")
        )

        for _, rnd in rounds_sorted.iterrows():
            rid = rnd["round_id"]
            # Snapshot BEFORE processing this round's heats
            snapshots[rid] = elo.snapshot()

            # Update ELO from heats in this round
            round_heats = hr[hr.round_id == rid]
            heat_names = round_heats["heat_title"].unique()
            for heat in heat_names:
                # Only use competitive heats (not qualifying draws)
                h_rows = round_heats[round_heats.heat_title == heat].copy()
                h_rows = h_rows.dropna(subset=["rank"])
                if len(h_rows) < 2:
                    continue
                results = [(r["rider_slug"], int(r["rank"])) for _, r in h_rows.iterrows()]
                results.sort(key=lambda x: x[1])
                elo.update_heat(results)

        return snapshots, elo

    # -----------------------------------------------------------------------
    def _clean_round_results(self, rr: pd.DataFrame) -> pd.DataFrame:
        df = rr.copy()
        df = df.rename(columns={
            "entry.object.slug": "rider_slug",
            "entry.object.firstName": "first_name",
            "entry.object.lastName": "last_name",
            "entry.object.country.code": "rider_country",
            "entry.isWildcard": "is_wildcard",
            "entry.isSubstitute": "is_substitute",
        })
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["points"] = pd.to_numeric(df["points"], errors="coerce")
        df["round_starts_at"] = pd.to_datetime(df["round_starts_at"], errors="coerce")
        df = df.dropna(subset=["rank", "rider_slug", "round_id"])
        df["is_wildcard"] = df.get("is_wildcard", False).fillna(False).astype(bool)
        df["is_substitute"] = df.get("is_substitute", False).fillna(False).astype(bool)
        # Apply slug alias normalisation (fixes corrupted API slugs like "-1" → "tai-woffinden")
        from ml.elo_seed import SLUG_ALIAS_MAP
        df["rider_slug"] = df["rider_slug"].map(lambda s: SLUG_ALIAS_MAP.get(s, s))
        return df

    def _clean_heat_rankings(self, hr: pd.DataFrame) -> pd.DataFrame:
        df = hr.copy()
        df = df.rename(columns={
            "entry.object.slug": "rider_slug",
            "title": "heat_title",
        })
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["points"] = pd.to_numeric(df["points"], errors="coerce")
        df = df.dropna(subset=["rider_slug"])
        # Apply slug alias normalisation (fixes corrupted API slugs like "-1" → "tai-woffinden")
        from ml.elo_seed import SLUG_ALIAS_MAP
        df["rider_slug"] = df["rider_slug"].map(lambda s: SLUG_ALIAS_MAP.get(s, s))
        return df

    def _clean_round_details(self, rd: pd.DataFrame) -> pd.DataFrame:
        df = rd[["round_id", "venue.slug", "venue.country.code",
                 "venue.trackLength", "venue.city"]].copy()
        df.columns = ["round_id", "venue_slug", "venue_country",
                      "track_length", "venue_city"]
        df["track_length"] = pd.to_numeric(df["track_length"], errors="coerce")
        df = df.drop_duplicates(subset=["round_id"])
        return df

    def _clean_riders(self, riders: pd.DataFrame) -> dict[str, dict]:
        """Build slug -> latest career stats dict."""
        result = {}
        numeric_cols = ["object.sgpWins", "object.appearances", "object.heatsWon",
                        "object.heatsRaced", "object.finals", "object.wins",
                        "object.fimRank"]
        for col in numeric_cols:
            if col in riders.columns:
                riders[col] = pd.to_numeric(riders[col], errors="coerce")

        for _, row in riders.iterrows():
            slug = row.get("object.slug") or row.get("slug")
            if pd.isna(slug):
                continue
            result[slug] = {
                "sgpWins": row.get("object.sgpWins"),
                "appearances": row.get("object.appearances"),
                "heatsWon": row.get("object.heatsWon"),
                "heatsRaced": row.get("object.heatsRaced"),
                "finals": row.get("object.finals"),
                "wins": row.get("object.wins"),
                "fimRank": row.get("object.fimRank"),
                "bornAt": row.get("object.bornAt"),
            }
        return result


# ---------------------------------------------------------------------------
# Feature column list (for inference alignment)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "elo_rating", "elo_minus_1500",
    "n_rounds_competed",
    "avg_rank_last5", "avg_heat_pts_last5",
    "min_rank_last5", "win_rate_last5", "top3_rate_last5",
    "last_round_rank", "last_heat_pts",
    "career_win_rate", "career_avg_rank",
    "venue_appearances", "venue_avg_rank", "venue_win",
    "track_length",
    "home_country",
    "is_wildcard", "is_substitute",
    "career_sgp_wins", "career_appearances",
    "career_heats_won", "career_heats_raced",
    "career_finals", "career_wins",
    "fim_rank", "career_heat_win_rate",
    "rider_age",
]
