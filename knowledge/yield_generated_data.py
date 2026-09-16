"""Utilities for the generated high-solid-content yield-stress dataset.

The source workbook uses Chinese process-column names and xlsx format. The rest
of the yield AutoML pipeline expects CSV with stable canonical names, a
normalized target column (`yield_stress`), and explicit data-fidelity metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from knowledge.yield_schema import GENERATED_PROCESS_COLUMN_MAP, INDEX_COLUMN, TARGET_COLUMN


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

GENERATED_COLUMN_MAP = GENERATED_PROCESS_COLUMN_MAP


def _cell_col_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", str(cell_ref).upper())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - ord("A") + 1)
    return max(0, value - 1)


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall("x:si", NS):
        values.append("".join(t.text or "" for t in item.findall(".//x:t", NS)))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//x:t", NS))
    value = cell.find("x:v", NS)
    if value is None:
        return ""
    raw = value.text or ""
    if cell_type == "s":
        return shared[int(raw)]
    try:
        return float(raw)
    except Exception:
        return raw


def read_xlsx_first_sheet(path: str | Path) -> pd.DataFrame:
    """Read the first worksheet without requiring openpyxl."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find(".//x:sheets/x:sheet", NS)
        if first_sheet is None:
            raise ValueError(f"No worksheets found in {path}")
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels:
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            target = "worksheets/sheet1.xml"
        sheet_path = "xl/" + target.lstrip("/")
        sheet = ET.fromstring(zf.read(sheet_path))

    rows: list[list[Any]] = []
    max_cols = 0
    for row in sheet.findall(".//x:sheetData/x:row", NS):
        values: list[Any] = []
        for cell in row.findall("x:c", NS):
            idx = _cell_col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = _cell_value(cell, shared)
        max_cols = max(max_cols, len(values))
        rows.append(values)
    if not rows:
        return pd.DataFrame()
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    header = [str(x).strip() for x in rows[0]]
    return pd.DataFrame(rows[1:], columns=header)


def read_generated_source_table(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read xlsx or CSV source tables used by the generated yield dataset."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return read_xlsx_first_sheet(path), {"reader": "xlsx_first_sheet"}
    if path.suffix.lower() == ".csv":
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(path, encoding=encoding), {
                    "reader": "csv",
                    "encoding": encoding,
                }
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
        raise ValueError(f"Could not read CSV {path}; tried utf-8/gb18030/gbk: {errors[:2]}")
    raise ValueError(f"Unsupported generated yield source format: {path.suffix}")


def normalize_generated_yield_dataframe(
    df: pd.DataFrame,
    *,
    data_fidelity: str = "augmented_hf",
    source: str = "generated_yield_stress_data",
    is_augmented: bool = True,
    sample_prefix: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize generated process-yield data to project columns."""
    missing = [col for col in ("屈服应力", "药浆温度", "锅内压力") if col not in df.columns]
    if missing:
        raise ValueError(f"Generated yield dataset is missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    prefix = sample_prefix or str(data_fidelity or "generated_yield")
    out[INDEX_COLUMN] = [f"{prefix}_{i:04d}" for i in range(1, len(df) + 1)]
    for src, dst in GENERATED_COLUMN_MAP.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    out["phi"] = 0.4
    out["data_fidelity"] = data_fidelity
    out["source"] = source
    out["is_augmented"] = bool(is_augmented)

    feature_columns = [
        col
        for col in out.columns
        if col not in {INDEX_COLUMN, TARGET_COLUMN, "data_fidelity", "source", "is_augmented"}
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    meta = {
        "source_schema": "generated_yield_process_202607",
        "raw_columns": list(df.columns),
        "column_map": {src: dst for src, dst in GENERATED_COLUMN_MAP.items() if src in df.columns},
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "target_unit": "Pa",
        "data_fidelity": data_fidelity,
        "is_augmented": bool(is_augmented),
        "assumptions": [
            "The source is process-control yield-stress data normalized from Chinese columns.",
            "phi is fixed to 0.4 to match the 2026-07-14 yield-stress notes.",
            "This step only normalizes schema; physics feature engineering is a separate next step.",
        ],
    }
    return out, meta


def prepare_generated_yield_dataset(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    data_fidelity: str = "augmented_hf",
    source: str = "generated_yield_stress_data",
    is_augmented: bool = True,
    output_name: str = "generated_yield_stress_data_normalized.csv",
    report_name: str = "generated_yield_stress_schema_report.json",
    sample_prefix: str | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, reader_meta = read_generated_source_table(input_path)
    normalized, meta = normalize_generated_yield_dataframe(
        raw,
        data_fidelity=data_fidelity,
        source=source,
        is_augmented=is_augmented,
        sample_prefix=sample_prefix,
    )
    csv_path = out_dir / output_name
    report_path = out_dir / report_name
    normalized.to_csv(csv_path, index=False, encoding="utf-8-sig")

    target = pd.to_numeric(normalized[TARGET_COLUMN], errors="coerce")
    report = {
        "input_path": str(input_path),
        "output_csv": str(csv_path),
        "n_rows": int(len(normalized)),
        "n_raw_columns": int(len(raw.columns)),
        "n_output_columns": int(len(normalized.columns)),
        **reader_meta,
        "target_summary": {
            "min": float(np.nanmin(target)),
            "max": float(np.nanmax(target)),
            "mean": float(np.nanmean(target)),
        },
        **meta,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv_path": str(csv_path), "report_path": str(report_path), "report": report}


def prepare_generated_yield_multifidelity_dataset(
    high_fidelity_path: str | Path,
    low_fidelity_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf = prepare_generated_yield_dataset(
        high_fidelity_path,
        out_dir,
        data_fidelity="high_fidelity",
        source="yield_stress_high_fidelity_csv",
        is_augmented=False,
        output_name="high_fidelity_normalized.csv",
        report_name="high_fidelity_schema_report.json",
        sample_prefix="hf",
    )
    lf = prepare_generated_yield_dataset(
        low_fidelity_path,
        out_dir,
        data_fidelity="low_fidelity",
        source="yield_stress_low_fidelity_csv",
        is_augmented=True,
        output_name="low_fidelity_normalized.csv",
        report_name="low_fidelity_schema_report.json",
        sample_prefix="lf",
    )

    hf_df = pd.read_csv(hf["csv_path"], encoding="utf-8-sig")
    lf_df = pd.read_csv(lf["csv_path"], encoding="utf-8-sig")
    combined = pd.concat([hf_df, lf_df], ignore_index=True)
    combined_path = out_dir / "multifidelity_normalized.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    target = pd.to_numeric(combined[TARGET_COLUMN], errors="coerce")
    report = {
        "input_high_fidelity_path": str(high_fidelity_path),
        "input_low_fidelity_path": str(low_fidelity_path),
        "high_fidelity_csv": hf["csv_path"],
        "low_fidelity_csv": lf["csv_path"],
        "combined_csv": str(combined_path),
        "n_high_fidelity": int(len(hf_df)),
        "n_low_fidelity": int(len(lf_df)),
        "n_total": int(len(combined)),
        "n_output_columns": int(len(combined.columns)),
        "fidelity_counts": {
            str(k): int(v)
            for k, v in combined["data_fidelity"].value_counts().to_dict().items()
        },
        "target_summary": {
            "min": float(np.nanmin(target)),
            "max": float(np.nanmax(target)),
            "mean": float(np.nanmean(target)),
        },
        "column_map": hf["report"].get("column_map", {}),
        "assumptions": [
            "The 20-row high-fidelity file is the trusted target-evaluation source.",
            "The 180-row low-fidelity file is retained as low_fidelity metadata, not silently treated as equivalent real HF.",
            "Current free-search can run on a single CSV; true multi-fidelity executor support remains a separate modeling step.",
        ],
    }
    report_path = out_dir / "multifidelity_schema_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "high_fidelity_csv": hf["csv_path"],
        "low_fidelity_csv": lf["csv_path"],
        "combined_csv": str(combined_path),
        "report_path": str(report_path),
        "report": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize generated yield-stress xlsx data.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--input-hf", default=None)
    parser.add_argument("--input-lf", default=None)
    parser.add_argument("--out-dir", default="agent_workspace/data/generated_yield_stress")
    parser.add_argument("--data-fidelity", default="augmented_hf")
    parser.add_argument("--source", default="generated_yield_stress_data")
    parser.add_argument("--real-hf", action="store_true")
    args = parser.parse_args(argv)
    if args.input_hf and args.input_lf:
        result = prepare_generated_yield_multifidelity_dataset(args.input_hf, args.input_lf, args.out_dir)
        print(json.dumps({
            "csv_path": result["combined_csv"],
            "high_fidelity_csv": result["high_fidelity_csv"],
            "low_fidelity_csv": result["low_fidelity_csv"],
            "report_path": result["report_path"],
        }, ensure_ascii=False))
    else:
        if not args.input:
            parser.error("--input is required unless both --input-hf and --input-lf are provided.")
        result = prepare_generated_yield_dataset(
            args.input,
            args.out_dir,
            data_fidelity=args.data_fidelity,
            source=args.source,
            is_augmented=not bool(args.real_hf),
        )
        print(json.dumps({"csv_path": result["csv_path"], "report_path": result["report_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
