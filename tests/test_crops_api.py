def test_create_crop_record_trims_text_and_returns_created_record(sqlite_api_client):
    response = sqlite_api_client.post(
        "/api/v1/crops",
        json={
            "crop_name": "  Soybean  ",
            "location": "  Indore, MP  ",
            "area_hectares": 2.5,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["crop_name"] == "Soybean"
    assert data["location"] == "Indore, MP"
    assert data["area_hectares"] == 2.5
    assert data["id"] > 0
    assert data["created_at"]


def test_list_crop_records_is_paginated(sqlite_api_client):
    for crop_name in ("Rice", "Wheat", "Soybean"):
        response = sqlite_api_client.post(
            "/api/v1/crops",
            json={
                "crop_name": crop_name,
                "location": "Indore",
                "area_hectares": 1,
            },
        )
        assert response.status_code == 201

    response = sqlite_api_client.get("/api/v1/crops?limit=2&offset=1")

    assert response.status_code == 200
    assert [record["crop_name"] for record in response.json()] == ["Wheat", "Rice"]


def test_list_crop_records_rejects_out_of_range_page_size(sqlite_api_client):
    response = sqlite_api_client.get("/api/v1/crops?limit=101")
    assert response.status_code == 422


def test_database_health_uses_database_session(sqlite_api_client):
    response = sqlite_api_client.get("/api/v1/db/health")
    assert response.status_code == 200
    assert response.json() == {"database": "connected"}
