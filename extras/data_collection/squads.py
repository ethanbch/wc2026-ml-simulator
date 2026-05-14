"""Scrape squads from cdm2026.fr and build squads.csv.

This script extracts player name, team, club, goals, caps, and position.
It maps French team names to the English names used by the project.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

URL = "https://www.cdm2026.fr/joueurs"
HEADERS = {"User-Agent": "Mozilla/5.0"}

POSITION_ALIASES = {
    "attaquants": "Attaquant",
    "milieux": "Milieu",
    "defenseurs": "Defenseur",
    "gardiens": "Gardien",
}

TEAM_MAPPING = {
    "Afrique du Sud": "South Africa",
    "Algérie": "Algeria",
    "Allemagne": "Germany",
    "Angleterre": "England",
    "Arabie saoudite": "Saudi Arabia",
    "Argentine": "Argentina",
    "Australie": "Australia",
    "Autriche": "Austria",
    "Belgique": "Belgium",
    "Brésil": "Brazil",
    "Canada": "Canada",
    "Cap-Vert": "Cabo Verde",
    "Colombie": "Colombia",
    "Corée du Sud": "Korea Republic",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Côte d’Ivoire": "Côte d'Ivoire",
    "Croatie": "Croatia",
    "Curaçao": "Curaçao",
    "Écosse": "Scotland",
    "Égypte": "Egypt",
    "Équateur": "Ecuador",
    "Espagne": "Spain",
    "Etats-Unis": "USA",
    "États-Unis": "USA",
    "France": "France",
    "Ghana": "Ghana",
    "Haïti": "Haiti",
    "Iran": "IR Iran",
    "Japon": "Japan",
    "Jordanie": "Jordan",
    "Maroc": "Morocco",
    "Mexique": "Mexico",
    "Norvège": "Norway",
    "Nouvelle-Zélande": "New Zealand",
    "Ouzbékistan": "Uzbekistan",
    "Panama": "Panama",
    "Paraguay": "Paraguay",
    "Pays-Bas": "Netherlands",
    "Portugal": "Portugal",
    "Qatar": "Qatar",
    "Sénégal": "Senegal",
    "Suisse": "Switzerland",
    "Tunisie": "Tunisia",
    "Uruguay": "Uruguay",
}

FLAG_PATTERN = re.compile(
    r"[\U0001F1E6-\U0001F1FF]+|\U0001F3F4[\U000E0060-\U000E007F]+"
)
PLAYER_PATTERN = re.compile(
    r"^(?P<player>.+?)\s+[\U0001F1E6-\U0001F1FF]{2}\s+(?P<team>.+?)\s+[·•]\s+"
    r"(?P<club>.+?)\s+(?P<goals>\d+)\s+but[s]?\s+(?P<caps>\d+)\s+(?:sel\.?|selections?)$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).lower()


def dedupe_team_name(name: str) -> str:
    tokens = name.split()
    if len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            return " ".join(tokens[:half])
    return name


def load_team_mapping(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        return {}
    df = pd.read_csv(mapping_path)
    if "source_name" not in df.columns or "team" not in df.columns:
        raise ValueError("team_name_mapping.csv must have columns: source_name, team")
    return dict(zip(df["source_name"].astype(str), df["team"].astype(str)))


def fetch_page(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_position_sections(soup: BeautifulSoup) -> list[tuple[str, list]]:
    headings = soup.find_all(["h2", "h3"])
    sections = []
    for idx, heading in enumerate(headings):
        heading_text = heading.get_text(" ", strip=True)
        normalized = normalize_text(heading_text)
        position = None
        for key, label in POSITION_ALIASES.items():
            if key in normalized:
                position = label
                break
        if not position:
            continue

        next_heading = None
        for next_idx in range(idx + 1, len(headings)):
            next_text = normalize_text(headings[next_idx].get_text(" ", strip=True))
            if any(key in next_text for key in POSITION_ALIASES):
                next_heading = headings[next_idx]
                break

        links = []
        for element in heading.next_elements:
            if element == next_heading:
                break
            if getattr(element, "name", None) == "a":
                href = element.get("href", "")
                if "/joueur/" in href:
                    links.append(element)
        sections.append((position, links))
    return sections


def parse_player_text(text: str) -> dict | None:
    text = text.replace("\u00a0", " ")
    text = " ".join(text.split())
    match = PLAYER_PATTERN.match(text)
    if not match:
        return None
    data = match.groupdict()
    data["team"] = dedupe_team_name(data["team"])
    data["goals"] = int(data["goals"])
    data["caps"] = int(data["caps"])
    return data


def _parse_int(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def parse_player_link(link) -> dict | None:
    p_texts = [p.get_text(" ", strip=True) for p in link.find_all("p")]
    if p_texts:
        player = p_texts[0]
        team_club = next((text for text in p_texts if "·" in text or "•" in text), None)
        goals_text = next((text for text in p_texts if "but" in text.lower()), None)
        caps_text = next((text for text in p_texts if "sel" in text.lower()), None)

        if team_club and goals_text and caps_text:
            team_club = FLAG_PATTERN.sub("", team_club)
            team_club = " ".join(team_club.replace("\u00a0", " ").split())
            parts = re.split(r"[·•]", team_club, maxsplit=1)
            if len(parts) == 2:
                team = dedupe_team_name(parts[0].strip())
                club = parts[1].strip()
                goals = _parse_int(goals_text)
                caps = _parse_int(caps_text)
                if goals is not None and caps is not None:
                    return {
                        "player": player,
                        "team": team,
                        "club": club,
                        "goals": goals,
                        "caps": caps,
                    }

    parsed = parse_player_text(link.get_text(" ", strip=True))
    if parsed:
        parsed["team"] = dedupe_team_name(parsed["team"])
        return parsed

    row = link.find_parent("tr")
    if row:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) >= 5:
            goals = _parse_int(cells[3])
            caps = _parse_int(cells[4])
            if goals is not None and caps is not None:
                team = FLAG_PATTERN.sub("", cells[1]).strip()
                return {
                    "player": cells[0],
                    "team": dedupe_team_name(team),
                    "club": cells[2],
                    "goals": goals,
                    "caps": caps,
                }

    return None


def build_rows(
    soup: BeautifulSoup,
    group_teams: set[str] | None,
    mapping: dict[str, str],
    only_groups: bool,
) -> list[dict]:
    rows = []
    unknown_teams = set()

    for position, links in find_position_sections(soup):
        for link in links:
            parsed = parse_player_link(link)
            if not parsed:
                continue

            team_raw = parsed["team"]
            team_mapped = mapping.get(team_raw, TEAM_MAPPING.get(team_raw, team_raw))
            if group_teams and team_mapped not in group_teams:
                unknown_teams.add(team_mapped)
                if only_groups:
                    continue

            rows.append(
                {
                    "team": team_mapped,
                    "player": parsed["player"],
                    "club": parsed["club"],
                    "position": position,
                    "minutes": parsed["caps"] * 90,
                    "caps": parsed["caps"],
                    "goals": parsed["goals"],
                }
            )

    if unknown_teams:
        logger.warning("Unknown teams encountered: %s", sorted(unknown_teams))
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape squads from cdm2026.fr")
    parser.add_argument("--groups-csv", default="data/raw/wc2026_groups.csv")
    parser.add_argument("--mapping-csv", default="data/raw/team_name_mapping.csv")
    parser.add_argument("--output", default="data/raw/squads.csv")
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include teams not in wc2026_groups.csv",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    groups_path = Path(args.groups_csv)
    group_teams = None
    if groups_path.exists():
        groups_df = pd.read_csv(groups_path)
        if "team" not in groups_df.columns:
            raise ValueError("wc2026_groups.csv must contain a 'team' column")
        group_teams = set(groups_df["team"].dropna().astype(str).tolist())
    else:
        logger.warning("Groups file not found: %s", groups_path)

    mapping = {}
    mapping_path = Path(args.mapping_csv)
    if mapping_path.exists():
        mapping = load_team_mapping(mapping_path)
        logger.info("Loaded %s team mappings", len(mapping))

    soup = fetch_page(URL)
    rows = build_rows(soup, group_teams, mapping, only_groups=not args.include_unknown)

    if not rows:
        raise RuntimeError("No players parsed. The site structure may have changed.")

    df = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %s rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
