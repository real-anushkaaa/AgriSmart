def test_versioned_api_requires_configured_api_key(sqlite_api_client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "api_access_token", "test-secret")

    unauthorized = sqlite_api_client.get("/api/v1/db/health")
    authorized = sqlite_api_client.get(
        "/api/v1/db/health",
        headers={"X-API-Key": "test-secret"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_production_refuses_versioned_api_without_api_key(sqlite_api_client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "app_env", "production")
    response = sqlite_api_client.get("/api/v1/db/health")

    assert response.status_code == 503
