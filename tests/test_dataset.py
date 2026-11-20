from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from instar import Task, dataset

if TYPE_CHECKING:
    from pathlib import Path


def test_from_jsonl_reads_tasks_with_a_custom_id_field(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text('{"case_id": 7, "prompt": "hi"}\n\n{"case_id": "b", "prompt": "yo"}\n')
    tasks = dataset.from_jsonl(path, id_field="case_id")
    assert [t.id for t in tasks] == ["7", "b"]
    assert tasks[0].prompt == "hi"
    assert tasks[0].data["case_id"] == 7


def test_from_csv_reads_rows_as_strings(tmp_path: Path) -> None:
    path = tmp_path / "set.csv"
    path.write_text("id,prompt,expected\na,hello,1\nb,bye,2\n")
    tasks = dataset.from_csv(path)
    assert [(t.id, t.prompt, t.expected) for t in tasks] == [("a", "hello", "1"), ("b", "bye", "2")]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"prompt": "no id"}\n', "set.jsonl:1: no 'id' field"),
        ('{"id": "a"}\n{"id": "a"}\n', "set.jsonl:2: duplicate task id 'a'"),
        ("[1, 2]\n", "set.jsonl:1: expected a JSON object"),
    ],
)
def test_from_jsonl_names_the_bad_line(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        dataset.from_jsonl(path)


def test_limit_and_missing_fields() -> None:
    tasks = [Task(id=str(i), data={"n": i}) for i in range(5)]
    assert [t.id for t in dataset.limit(tasks, 2)] == ["0", "1"]
    assert len(dataset.limit(tasks, None)) == 5
    with pytest.raises(AttributeError, match="task '0' has no field 'prompt'"):
        _ = tasks[0].prompt
