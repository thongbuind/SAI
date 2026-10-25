"""Phòng làm việc (``python -m web --lab``, /api/lab): chặn Host lạ, kiểm spec, hàng đợi job, SSE, huỷ, đọc run,
ảnh chụp cho bản static.

Phần lớn test dùng một "lab giả" (script Python nhỏ ghi đúng định dạng thư mục run) để chạy nhanh và
điều khiển được kết quả. Hai test cuối gọi lệnh lab thật để kiểm hợp đồng giữa hai bên.
"""

import asyncio
import json
import sys
import time
import pytest
import torch
from fastapi.testclient import TestClient
from helpers import INFER_MODEL
from test_web import fake_loader, isolated_data  # noqa: F401  (fixture dùng chung với test web)
from foundation_lab.catalog import build_model
from web import __main__ as cli
from web.backend.lab_cli import LAB_COMMAND, describe
from web.backend.lab_jobs import MAX_QUEUE, JobRunner
from web.backend.lab_routes import Lab, host_name
from web.backend.lab_runs import read_events
from web.backend.lab_snapshot import LabSnapshot
from web.backend.main import create_app
from web.backend.model_registry import ModelRegistry
from web.backend.paths import PROJECT_ROOT

LOCAL_URL = "http://127.0.0.1"
WAIT_TIMEOUT_SEC = 30
POLL_SEC = 0.05

# Lab giả: đọc spec từ stdin, hành vi chọn bằng spec["fake"]; ghi thư mục run như lệnh infer thật.
FAKE_LAB = r'''
import json, sys, time
from pathlib import Path
args = sys.argv[1:]
command, run_id, runs_root = args[0], args[args.index("--run-id") + 1], args[args.index("--runs-root") + 1]
spec = json.load(sys.stdin)
if spec["fake"] == "config_error":
    print("Lỗi config: prompts rỗng", file=sys.stderr)
    sys.exit(2)
run = Path(runs_root) / command / run_id
run.mkdir(parents=True)
(run / "resolved_config.yaml").write_text(json.dumps(spec), encoding="utf-8")
def emit(seq, kind, **data):
    with open(run / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": seq, "time": time.time(), "kind": kind, **data}) + "\n")
emit(1, "started", total=2)
if spec["fake"] == "slow":
    time.sleep(60)
emit(2, "progress", done=1, total=2)
if spec["fake"] == "fail":
    emit(3, "failed", error="RuntimeError: hỏng")
    sys.exit(1)
emit(3, "progress", done=2, total=2)
result = {"format_version": 1, "command": command, "metrics": [], "records": [{"text": "ok"}], "timings": {}}
(run / "result.json").write_text(json.dumps(result), encoding="utf-8")
emit(4, "finished")
if spec["fake"] == "linger":
    time.sleep(2)
'''

DESCRIPTION = {
    "format_version": 1,
    "commands": {
        "infer": {"department": "inference", "spec": {"type": "object", "fields": {}}},
        "compare-kernel": {"department": "arch", "spec": {"type": "object", "fields": {}}},
        "compare-objective": {"department": "objectives", "spec": {"type": "object", "fields": {}}},
        # Lệnh lab mà web chưa có kiểm tra spec: web không chạy nó.
        "unchecked": {"department": "eval", "spec": {"type": "object", "fields": {}}},
    },
    "models": ["SAI_35M", "SAI_375M", "SAI_modular_35M"],
    "model_families": {"SAI_35M": "SAI", "SAI_375M": "SAI", "SAI_modular_35M": "SAI_modular"},
    "experiments": {"SAI_35M": ["pretrain", "sft1"]},
    "weights": ["model/sft1_35M.pt", "model/sft1_375M.pt"],
    "checkpoints": {"SAI_35M": ["model/sft1_35M.pt"], "SAI_375M": ["model/sft1_375M.pt"]},
    "benchmark_path": "benchmark.yaml",
}
BENCHMARK_HEADER = "# Bộ câu hỏi mặc định.\n# Dòng comment này phải còn sau khi web ghi.\n"

def fake_spec(behaviour: str) -> dict:
    return {"model": "SAI_35M", "checkpoint": "model/sft1_35M.pt", "fake": behaviour}

@pytest.fixture
def runs_root(tmp_path):
    return tmp_path / "runs"

@pytest.fixture
def site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>sai</h1>", encoding="utf-8")
    return site

@pytest.fixture
def client(tmp_path, runs_root, site, isolated_data):
    script = tmp_path / "fake_lab.py"
    script.write_text(FAKE_LAB, encoding="utf-8")
    (tmp_path / DESCRIPTION["benchmark_path"]).write_text(BENCHMARK_HEADER + "prompts:\n- xin chào\n", encoding="utf-8")
    lab = Lab(JobRunner((sys.executable, str(script)), tmp_path, runs_root), DESCRIPTION, runs_root)
    app = create_app(site, ModelRegistry(["SAI_35M"], fake_loader), lab)
    with TestClient(app, base_url=LOCAL_URL) as test_client:
        yield test_client

def submit(client, behaviour: str) -> str:
    response = client.post("/api/lab/jobs", json={"command": "infer", "spec": fake_spec(behaviour)})
    assert response.status_code == 202, response.text
    return response.json()["run_id"]

def wait_state(client, run_id: str, states: tuple[str, ...], command: str = "infer") -> dict:
    deadline = time.monotonic() + WAIT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        detail = client.get(f"/api/lab/jobs/{command}/{run_id}").json()
        if detail["state"] in states:
            return detail
        time.sleep(POLL_SEC)
    raise AssertionError(f"run {run_id} không tới trạng thái {states}; đang là {detail['state']}")

def sse_events(client, run_id: str) -> list[tuple[str, dict]]:
    events = []
    with client.stream("GET", f"/api/lab/jobs/infer/{run_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        kind = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((kind, json.loads(line.removeprefix("data: "))))
    return events

@pytest.mark.parametrize("header, name", [("127.0.0.1:8000", "127.0.0.1"), ("[::1]:8000", "[::1]"), ("localhost", "localhost")])
def test_host_name(header, name):
    assert host_name(header) == name

def test_lab_only_answers_localhost(client):
    assert client.get("/api/lab/describe").json() == DESCRIPTION
    assert client.get("/api/lab/describe", headers={"host": "evil.example:8000"}).status_code == 403
    # Chat và site vẫn phục vụ mọi Host như trước (deploy sau tên miền thật).
    assert client.get("/", headers={"host": "sai.example"}).text == "<h1>sai</h1>"
    assert client.get("/api/models", headers={"host": "sai.example"}).status_code == 200

def test_lab_disabled_returns_503_with_reason(site, isolated_data):
    client = TestClient(create_app(site, ModelRegistry(["SAI_35M"], fake_loader)), base_url=LOCAL_URL)
    for response in (client.get("/api/lab/describe"), client.post("/api/lab/jobs", json={})):
        assert response.status_code == 503
        assert "python -m web --lab" in response.json()["detail"]

def test_lab_flag_requires_loopback_host():
    assert cli.parse_args(["--lab"]).lab
    with pytest.raises(SystemExit):
        cli.parse_args(["--lab", "--host", "0.0.0.0"])

@pytest.mark.parametrize("body", [
    {"command": "train", "spec": fake_spec("ok")},
    {"command": "infer", "spec": {**fake_spec("ok"), "checkpoint": "/etc/passwd"}},
    {"command": "infer", "spec": {**fake_spec("ok"), "model": {"family": "SAI"}}},
    # Trọng số có trên máy nhưng của model khác cỡ.
    {"command": "infer", "spec": {**fake_spec("ok"), "checkpoint": "model/sft1_375M.pt"}},
    # Tên model thành đường dẫn file config trong lab: chỉ nhận tên lab đã liệt kê, không nhận config viết sẵn.
    {"command": "compare-kernel", "spec": {"candidate": "eager", "models": ["../../etc/passwd"]}},
    {"command": "compare-kernel", "spec": {"candidate": "eager", "models": {"x": {"family": "SAI_modular"}}}},
    {"command": "compare-kernel", "spec": {"candidate": "eager"}},
    # Tên experiment cũng thành đường dẫn file trong lab.
    {"command": "compare-objective", "spec": {"candidate": "lm_ce", "experiments": ["../../etc/passwd"]}},
    {"command": "compare-objective", "spec": {"candidate": "lm_ce", "experiments": {"x": {"objective": "lm_ce"}}}},
    {"command": "compare-objective", "spec": {"candidate": "lm_ce"}},
    {"command": "unchecked", "spec": {"fake": "ok"}},
])
def test_submit_rejects_unknown_command_and_unlisted_resources(client, body):
    assert client.post("/api/lab/jobs", json=body).status_code == 400

def test_compare_kernel_job_runs_with_listed_models(client, runs_root):
    spec = {"candidate": "eager", "models": ["SAI_modular_35M"], "fake": "ok"}
    response = client.post("/api/lab/jobs", json={"command": "compare-kernel", "spec": spec})
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    assert wait_state(client, run_id, ("succeeded",), command="compare-kernel")["config"]["models"] == ["SAI_modular_35M"]
    assert (runs_root / "compare-kernel" / run_id / "result.json").exists()

def test_compare_objective_job_runs_with_listed_experiments(client, runs_root):
    spec = {"candidate": "lm_ce", "experiments": ["SAI_35M"], "stage": "sft1", "fake": "ok"}
    response = client.post("/api/lab/jobs", json={"command": "compare-objective", "spec": spec})
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    assert wait_state(client, run_id, ("succeeded",), command="compare-objective")["config"]["experiments"] == ["SAI_35M"]
    assert (runs_root / "compare-objective" / run_id / "result.json").exists()

def test_benchmark_read_and_save_keeps_header(client, tmp_path):
    assert client.get("/api/lab/benchmark").json() == {"prompts": ["xin chào"]}
    response = client.put("/api/lab/benchmark", json={"prompts": [" thủ đô của việt nam là gì? ", "bts có mấy thành viên"]})
    assert response.json() == {"prompts": ["thủ đô của việt nam là gì?", "bts có mấy thành viên"]}
    assert client.get("/api/lab/benchmark").json() == response.json()
    assert (tmp_path / DESCRIPTION["benchmark_path"]).read_text(encoding="utf-8").startswith(BENCHMARK_HEADER)

@pytest.mark.parametrize("prompts", [[], ["a", "  "]])
def test_benchmark_rejects_empty_prompts(client, tmp_path, prompts):
    before = (tmp_path / DESCRIPTION["benchmark_path"]).read_text(encoding="utf-8")
    assert client.put("/api/lab/benchmark", json={"prompts": prompts}).status_code == 400
    assert (tmp_path / DESCRIPTION["benchmark_path"]).read_text(encoding="utf-8") == before

def test_bad_run_id_is_rejected(client):
    assert client.get("/api/lab/jobs/infer/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/api/lab/jobs/infer/a.b").status_code == 422

def test_successful_job_is_listed_with_result(client):
    run_id = submit(client, "ok")
    detail = wait_state(client, run_id, ("succeeded",))
    assert detail["result"]["records"] == [{"text": "ok"}]
    assert [event["kind"] for event in detail["events"]] == ["started", "progress", "progress", "finished"]
    assert detail["config"]["fake"] == "ok"
    runs = client.get("/api/lab/jobs", params={"command": "infer"}).json()["runs"]
    assert [(run["run_id"], run["state"]) for run in runs] == [(run_id, "succeeded")]

def test_events_stream_ends_with_final_state(client):
    run_id = submit(client, "ok")
    events = sse_events(client, run_id)
    assert [kind for kind, _ in events] == ["started", "progress", "progress", "finished", "end"]
    assert events[-1][1] == {"state": "succeeded"}

def test_finished_run_is_final_while_process_still_exits(client):
    """Lab ghi "finished" rồi mới thoát; trong khoảng đó trạng thái phải là succeeded để trang không mở lại luồng."""
    run_id = submit(client, "linger")
    events = sse_events(client, run_id)
    assert events[-1] == ("end", {"state": "succeeded"})
    assert client.get(f"/api/lab/jobs/infer/{run_id}").json()["state"] == "succeeded"

def test_events_stream_resumes_after_last_event_id(client):
    run_id = submit(client, "ok")
    wait_state(client, run_id, ("succeeded",))
    with client.stream("GET", f"/api/lab/jobs/infer/{run_id}/events", headers={"Last-Event-ID": "3"}) as response:
        kinds = [line.removeprefix("event: ") for line in response.iter_lines() if line.startswith("event: ")]
    assert kinds == ["finished", "end"]

def test_runtime_failure_and_config_error(client):
    failed = submit(client, "fail")
    assert wait_state(client, failed, ("failed",))["events"][-1]["error"] == "RuntimeError: hỏng"
    # Lỗi config: lab thoát mã 2 trước khi tạo thư mục run; lý do chỉ có trong stderr mà server giữ.
    config_error = submit(client, "config_error")
    detail = wait_state(client, config_error, ("failed",))
    assert detail["events"] == [] and "Lỗi config" in detail["stderr"][-1]

def test_cancel_running_job(client, runs_root):
    run_id = submit(client, "slow")
    wait_state(client, run_id, ("running",))
    deadline = time.monotonic() + WAIT_TIMEOUT_SEC
    while not (runs_root / "infer" / run_id / "events.jsonl").exists() and time.monotonic() < deadline:
        time.sleep(POLL_SEC)
    assert client.post(f"/api/lab/jobs/infer/{run_id}/cancel").json() == {"state": "cancelled"}
    assert client.get(f"/api/lab/jobs/infer/{run_id}").json()["state"] == "cancelled"
    assert client.post(f"/api/lab/jobs/infer/{run_id}/cancel").status_code == 409

def test_queue_is_bounded(client):
    running = submit(client, "slow")
    wait_state(client, running, ("running",))
    queued = [submit(client, "ok") for _ in range(MAX_QUEUE)]
    response = client.post("/api/lab/jobs", json={"command": "infer", "spec": fake_spec("ok")})
    assert response.status_code == 503
    # Job đang chờ bị huỷ thì không bao giờ chạy.
    client.post(f"/api/lab/jobs/infer/{queued[0]}/cancel")
    client.post(f"/api/lab/jobs/infer/{running}/cancel")
    assert wait_state(client, queued[0], ("cancelled",))["events"] == []
    for run_id in queued[1:]:
        wait_state(client, run_id, ("succeeded",))

def test_runs_written_outside_the_server_are_read_from_disk(client, runs_root):
    """Run chạy bằng CLI (không qua web), hoặc trước khi server khởi động lại: trạng thái suy ra từ file."""
    def write_run(run_id, events, result=None):
        path = runs_root / "infer" / run_id
        path.mkdir(parents=True)
        (path / "resolved_config.yaml").write_text("prompts: [a]\n", encoding="utf-8")
        (path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        if result is not None:
            (path / "result.json").write_text(json.dumps(result), encoding="utf-8")

    write_run("20261020-100000", [{"seq": 1, "kind": "started"}], {"format_version": 1, "records": []})
    write_run("20261021-100000", [{"seq": 1, "kind": "started"}, {"seq": 2, "kind": "failed", "error": "x"}])
    write_run("20261022-100000", [{"seq": 1, "kind": "started"}])
    runs = client.get("/api/lab/jobs", params={"command": "infer"}).json()["runs"]
    assert [(run["run_id"], run["state"]) for run in runs] == [
        ("20261022-100000", "incomplete"), ("20261021-100000", "failed"), ("20261020-100000", "succeeded"),
    ]
    assert runs[0]["config"] == {"prompts": ["a"]}
    assert sse_events(client, "20261022-100000")[-1] == ("end", {"state": "incomplete"})

def test_snapshot_has_every_run_on_disk_like_the_api(client, tmp_path, runs_root):
    """Bản static đọc ảnh chụp thay cho API: mỗi run giống GET /api/lab/jobs/<lệnh>/<run_id>, trừ stderr (chỉ job trong
    bộ nhớ mới có); lệnh chưa có run vẫn có khoá với danh sách rỗng."""
    ok_id = submit(client, "ok")
    wait_state(client, ok_id, ("succeeded",))
    fail_id = submit(client, "fail")
    wait_state(client, fail_id, ("failed",))
    out_path = tmp_path / "lab_snapshot.json"
    LabSnapshot(DESCRIPTION, tmp_path, runs_root, out_path).write()

    snapshot = json.loads(out_path.read_text(encoding="utf-8"))
    assert snapshot["description"] == DESCRIPTION
    assert snapshot["benchmark"] == ["xin chào"]
    assert snapshot["runs"].keys() == DESCRIPTION["commands"].keys()
    assert snapshot["runs"]["compare-kernel"] == []
    assert [run["run_id"] for run in snapshot["runs"]["infer"]] == [fail_id, ok_id]
    for run in snapshot["runs"]["infer"]:
        detail = client.get(f"/api/lab/jobs/infer/{run['run_id']}").json()
        assert run == {**detail, "stderr": []}

def test_runner_calls_after_job_and_keeps_running_when_it_fails(tmp_path, runs_root):
    """after_job (chụp ảnh cho bản static) chạy sau mỗi job; lỗi trong đó không làm kẹt các job sau."""
    script = tmp_path / "fake_lab.py"
    script.write_text(FAKE_LAB, encoding="utf-8")
    calls = []

    def after_job():
        calls.append(len(calls))
        if len(calls) == 1:
            raise OSError("đĩa đầy")

    async def run_jobs():
        runner = JobRunner((sys.executable, str(script)), tmp_path, runs_root, after_job=after_job)
        runner.start()
        jobs = [runner.submit("infer", fake_spec("ok")), runner.submit("infer", fake_spec("fail"))]
        deadline = time.monotonic() + WAIT_TIMEOUT_SEC
        while len(calls) < len(jobs) and time.monotonic() < deadline:
            await asyncio.sleep(POLL_SEC)
        await runner.stop()
        return jobs

    jobs = asyncio.run(run_jobs())
    assert [job.state for job in jobs] == ["succeeded", "failed"]
    assert calls == [0, 1]

def test_read_events_skips_partial_last_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq": 1, "kind": "started"}\n{"seq": 2, "ki', encoding="utf-8")
    events, offset = read_events(path)
    assert [event["seq"] for event in events] == [1]
    with open(path, "a", encoding="utf-8") as f:
        f.write('nd": "progress"}\n')
    events, _ = read_events(path, offset)
    assert events == [{"seq": 2, "kind": "progress"}]

def test_real_lab_describe_matches_web_format():
    description = describe(LAB_COMMAND, PROJECT_ROOT)
    assert {"infer", "compare-kernel", "compare-objective"} <= set(description["commands"])
    assert {"checkpoints", "benchmark_path", "departments", "models", "model_families", "experiments"} <= set(description)
    assert set(description["model_families"]) == set(description["models"])
    assert (PROJECT_ROOT / description["benchmark_path"]).is_file()

def test_lab_fields_match_describe_schema():
    """src/data/lab_fields.json (nhãn, điều kiện hiện của form) chỉ được nói về khoá có thật trong schema của lab."""
    description = describe(LAB_COMMAND, PROJECT_ROOT)
    hints_by_command = json.loads((PROJECT_ROOT / "web" / "site" / "src" / "data" / "lab_fields.json").read_text(encoding="utf-8"))
    assert set(hints_by_command) <= set(description["commands"])

    def field_at(schema, path):
        """``variants[].generation.strategy`` → mô tả của khoá đó trong schema."""
        node = schema
        for part in path.split("."):
            name = part.removesuffix("[]")
            node = node["fields"][name]
            if part.endswith("[]"):
                node = node["items"]
        return node

    for command, hints in hints_by_command.items():
        spec = description["commands"][command]["spec"]
        for path, hint in hints.items():
            field = field_at(spec, path)
            for sibling in hint.get("visible_when", {}):
                parent = path.rsplit(".", 1)[0] if "." in path else ""
                field_at(spec, f"{parent}.{sibling}" if parent else sibling)
            source = hint.get("source")
            if source is not None and source.startswith("models:"):
                # "models:<họ>": tên model thuộc họ đó, theo "model_families" của describe.
                family = source.removeprefix("models:")
                choices = [name for name, model_family in description["model_families"].items() if model_family == family]
                assert choices and set(hint.get("initial", [])) <= set(choices), path
            elif source is not None and not source.startswith("registry:"):
                assert source in description, path
            if "filter_by" in hint:
                parent = path.rsplit(".", 1)[0] if "." in path else ""
                field_at(spec, f"{parent}.{hint['filter_by']}" if parent else hint["filter_by"])
            if "initial" in hint and hint.get("source") in description:
                choices = description[hint["source"]]
                if "filter_by" in hint:
                    choices = [value for values in choices.values() for value in values]
                # Chips có giá trị ban đầu là danh sách lựa chọn đang bật.
                initial = hint["initial"] if isinstance(hint["initial"], list) else [hint["initial"]]
                assert set(initial) <= set(choices) or not choices, path
            assert not (field["type"] in ("map", "union", "any") and "source" not in hint and not hint.get("hidden")), \
                f"{path}: kiểu {field['type']} cần 'source' hoặc 'hidden' để form có ô nhập"
            widget = hint.get("widget")
            if widget == "slider":
                assert field["type"] in ("int", "float") and {"min", "max"} <= set(hint), f"{path}: slider cần số và min/max"
            if widget == "segmented":
                assert field["type"] == "choice" or "choices" in hint or "source" in hint, f"{path}: segmented cần danh sách lựa chọn"
            if widget == "lines":
                assert field["type"] == "list" and field["items"]["type"] == "str", f"{path}: lines cần list[str]"
            if widget == "chips":
                assert source is not None, f"{path}: chips cần 'source' để có danh sách lựa chọn"
            if hint.get("baseline"):
                assert field["type"] == "list" and field["items"]["type"] == "object", f"{path}: baseline chỉ dùng cho danh sách object"
            if "summary" in hint:
                assert "{}" in hint["summary"], f"{path}: mẫu tóm tắt cần chỗ '{{}}' cho giá trị"
            if hint.get("inline"):
                assert field["type"] == "object", f"{path}: inline chỉ dùng cho object lồng"
            assert hint.get("group") in (None, "advanced") and hint.get("width") in (None, "half"), path

def test_real_lab_infer_through_runner(tmp_path):
    """Hợp đồng thật giữa hai bên: spec qua stdin, --run-id/--runs-root, mã thoát, file trong thư mục run."""
    torch.manual_seed(0)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(build_model(INFER_MODEL).state_dict(), checkpoint)
    spec = {
        "model": INFER_MODEL, "checkpoint": str(checkpoint), "device": "cpu", "prompts": ["xin chào"],
        "variants": [{"name": "default"}],
    }

    async def run_job():
        runner = JobRunner(LAB_COMMAND, PROJECT_ROOT, tmp_path / "runs")
        runner.start()
        job = runner.submit("infer", spec)
        while job.state in ("queued", "running"):
            await asyncio.sleep(POLL_SEC)
        await runner.stop()
        return job

    job = asyncio.run(run_job())
    assert job.state == "succeeded", list(job.stderr_tail)
    run_path = tmp_path / "runs" / "infer" / job.run_id
    result = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    assert result["format_version"] == 1 and len(result["records"]) == 1
    events, _ = read_events(run_path / "events.jsonl")
    assert events[-1]["kind"] == "finished"
    # python -m đặt argv[0] tuyệt đối; lab ghi lại tương đối theo gốc dự án. runs_root nằm ngoài dự án nên giữ nguyên.
    invocation = json.loads((run_path / "invocations.jsonl").read_text(encoding="utf-8"))
    assert invocation["argv"][0] == "foundation_lab/__main__.py"
    assert invocation["argv"][invocation["argv"].index("--runs-root") + 1] == str(tmp_path / "runs")
