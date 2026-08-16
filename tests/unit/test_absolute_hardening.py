from fastapi import HTTPException
from ik_kernel.app import create_app
from ik_kernel.config import Settings
from ik_kernel.deps import get_current_principal
from ik_indus_llm.tokenizer import IndusTokenizer
from ik_planning import Plan,PlanStep
from ik_router.budget import BudgetEnforcer,BudgetExceededError

def test_settings_are_injected_not_global():
    s=Settings(environment="production",debug=False,api_keys="kid:secret:tenant:admin:*",jwt_secret="x"*64,api_cors_origins=["https://example.com"],api_allowed_hosts=["example.com"],neo4j_password="secure")
    app=create_app(s); assert app.state.settings is s
    try:get_current_principal(settings=s)
    except HTTPException as e: assert e.status_code==401
    else: raise AssertionError("production configuration allowed anonymous access")

def test_char_tokenizer_requires_explicit_fit():
    t=IndusTokenizer("char")
    try:t.encode("x")
    except RuntimeError:pass
    else:raise AssertionError("unfitted tokenizer encoded")
    t.fit("abc"); ids=t.encode("abc"); assert t.decode(ids)=="abc" and ids==t.encode("abc")

def test_planner_rejects_cycles():
    try:Plan("x",[PlanStep("a","A",["b"]),PlanStep("b","B",["a"])]).validate()
    except ValueError as e:assert "cycle" in str(e)
    else:raise AssertionError("cycle accepted")

def test_budget_reservation_is_atomic():
    b=BudgetEnforcer();b.set_budget("t",10,10);assert b.reserve("t",10,10)==(10,10)
    try:b.reserve("t",1,1)
    except BudgetExceededError:pass
    else:raise AssertionError("budget overrun accepted")


def test_application_scoped_settings_are_used_by_health_endpoints():
    from fastapi.testclient import TestClient
    from ik_kernel.app import create_app
    from ik_kernel.config import Settings

    settings = Settings(environment="test", default_tenant_id="scoped-tenant", api_prefix="/scoped")
    app = create_app(settings)
    with TestClient(app) as client:
        version = client.get("/version")
        assert version.status_code == 200
        assert version.json()["environment"] == "test"
        assert version.json()["api_prefix"] == "/scoped"
