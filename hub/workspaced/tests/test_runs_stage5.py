from app.schemas import RunConfig, RunStartResponse


def test_run_config_carries_restart_descriptor():
    config = RunConfig(
        name="web",
        cmd="npm run dev",
        long_running=True,
        cwd="/workspace/frontend",
        env={"NODE_ENV": "development"},
        ports=[3000],
        resume_policy="restart",
        readiness={"timeout_sec": 15},
    )

    assert config.cwd == "/workspace/frontend"
    assert config.resume_policy == "restart"
    assert config.ports == [3000]


def test_run_start_response_defaults_are_backward_compatible():
    response = RunStartResponse(name="test", long_running=False, exit_code=0)

    assert response.resume_policy == "never"
    assert response.cwd == "/workspace"
    assert response.env == {}
    assert response.ports == []
