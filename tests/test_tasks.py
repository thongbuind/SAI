"""Hợp đồng tác vụ chung (``foundation_lab.core.tasks``): mọi tác vụ có lệnh con cùng dạng và có mặt trong ``describe``."""

import json
import pytest
import foundation_lab
from foundation_lab.catalog import TASKS
from foundation_lab.cli import build_parser, main
from web.scripts.export_site_data import DEPARTMENT_LINE

# Mã phòng lấy từ docstring của lab, cùng cách web dựng danh sách phòng.
DEPARTMENT_CODES = {match["code"] for match in DEPARTMENT_LINE.finditer(foundation_lab.__doc__)}

@pytest.mark.parametrize("name", TASKS.names())
def test_task_command_takes_the_shared_arguments(name):
    args = build_parser().parse_args([name, "-", "--set", "a=1", "--runs-root", "r", "--run-id", "x"])
    assert (args.spec, args.set, args.runs_root, args.run_id) == ("-", ["a=1"], "r", "x")
    # Spec bỏ trống: chỉ dùng --set.
    assert build_parser().parse_args([name]).spec is None

def test_describe_lists_every_task_with_its_department(capsys):
    main(["describe"])
    commands = json.loads(capsys.readouterr().out)["commands"]
    assert set(commands) == set(TASKS.names())
    for name, command in commands.items():
        assert command["department"] in DEPARTMENT_CODES, name
        assert command["spec"]["type"] == "object" and command["spec"]["fields"], name
