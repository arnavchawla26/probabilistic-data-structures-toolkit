import json

import pytest

from pdst.cli import main, read_tokens


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_tokens_skips_blank_lines(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("a\n\nb\n\n\nc\n", encoding="utf-8")
    assert read_tokens(str(path)) == ["a", "b", "c"]


def test_read_tokens_with_csv_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,name\n1,alice\n2,bob\n", encoding="utf-8")
    assert read_tokens(str(path), column=1) == ["name", "alice", "bob"]


def test_read_tokens_column_out_of_range_raises(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_tokens(str(path), column=5)


def test_membership_command_reports_no_false_negatives(tmp_path, capsys):
    items = [f"user-{i}@example.com" for i in range(500)]
    input_path = tmp_path / "items.txt"
    _write_lines(input_path, items)

    exit_code = main(["membership", "--input", str(input_path), "--fp-rate", "0.01", "--json"])
    assert exit_code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["structure"] == "bloom_filter"
    assert result["distinct_items_added"] == 500
    assert result["false_negatives_should_always_be_zero"] == 0


def test_membership_command_with_explicit_queries_file(tmp_path, capsys):
    input_path = tmp_path / "items.txt"
    _write_lines(input_path, ["apple", "banana", "cherry"])
    queries_path = tmp_path / "queries.txt"
    _write_lines(queries_path, ["apple", "durian", "banana", "elderberry"])

    exit_code = main(["membership", "--input", str(input_path), "--queries", str(queries_path), "--json"])
    assert exit_code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["queries_tested"] == 4
    assert result["true_positives"] == 2  # apple, banana
    assert result["false_negatives_should_always_be_zero"] == 0


def test_frequency_command_top_k_matches_exact_ranking(tmp_path, capsys):
    tokens = ["a"] * 50 + ["b"] * 30 + ["c"] * 10 + ["d"] * 5
    input_path = tmp_path / "stream.txt"
    _write_lines(input_path, tokens)

    exit_code = main(["frequency", "--input", str(input_path), "--top-k", "3", "--json"])
    assert exit_code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["structure"] == "count_min_sketch"
    assert result["total_tokens_processed"] == len(tokens)
    top_items = [row["item"] for row in result["top_k"]]
    assert top_items == ["a", "b", "c"]
    assert result["all_estimates_within_error_bound"] is True


def test_cardinality_command_is_close_to_exact(tmp_path, capsys):
    tokens = [f"visitor-{i}" for i in range(5000)]
    input_path = tmp_path / "stream.txt"
    _write_lines(input_path, tokens)

    exit_code = main(["cardinality", "--input", str(input_path), "--precision", "12", "--json"])
    assert exit_code == 0

    result = json.loads(capsys.readouterr().out)
    assert result["structure"] == "hyperloglog"
    assert result["exact_distinct_count"] == 5000
    assert result["relative_error"] < 0.1


def test_text_report_mode_does_not_crash(tmp_path, capsys):
    input_path = tmp_path / "stream.txt"
    _write_lines(input_path, ["x", "y", "z", "x", "y", "x"])

    exit_code = main(["frequency", "--input", str(input_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Count-Min Sketch" in out
    assert "top_k:" in out


def test_missing_input_file_returns_error_exit_code(capsys):
    exit_code = main(["cardinality", "--input", "/nonexistent/path/does-not-exist.txt"])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])
